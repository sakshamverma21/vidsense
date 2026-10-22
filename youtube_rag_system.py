import re
from urllib.parse import urlparse, parse_qs
from typing import List, Dict, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain_community.embeddings import FastEmbedEmbeddings
from transformers import BertTokenizer, BertModel
import torch
from sklearn.cluster import KMeans

import os
from dotenv import load_dotenv

# Load GROQ_API_KEY (and any other secrets) from a .env file at the project root.
# NOTE: previously this line blanked the key (os.environ["GROQ_API_KEY"] = ""),
# which silently broke the Groq backend. We now load real values from .env.
load_dotenv()


class BERTTopicExtractor:
    """BERT-based topic extraction and clustering for video segments"""
    
    def __init__(self):
        print("🔄 Loading BERT model (first time may take a minute)...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        print("✅ BERT model loaded successfully")
    
    def get_bert_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate BERT embeddings for text segments"""
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize and prepare input
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    max_length=512, 
                    truncation=True,
                    padding=True
                )
                
                # Get BERT output
                outputs = self.model(**inputs)
                
                # Use [CLS] token embedding as sentence representation
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
                embeddings.append(cls_embedding.numpy())
        
        return np.array(embeddings)
    
    def cluster_topics(self, texts: List[str], n_clusters: int = 5) -> Dict:
        """Cluster text segments into topics using BERT embeddings and K-Means"""
        print(f"\n🤖 Running BERT-based topic clustering...")
        
        # Get BERT embeddings
        embeddings = self.get_bert_embeddings(texts)
        
        # Perform K-Means clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(texts)), random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Organize results
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'text': texts[idx],
                'index': idx
            })
        
        print(f"✅ Identified {len(clusters)} topic clusters using BERT")
        
        return {
            'clusters': clusters,
            'embeddings': embeddings,
            'labels': cluster_labels
        }


class YouTubeRAGSystem:
    """Enhanced YouTube Video Analysis System with RAG, BERT, and dual LLM support"""
    
    def __init__(self, model_type: str = "groq", model_name: str = "llama-3.3-70b-versatile"):
        """
        Initialize the RAG system with choice of LLM backend
        
        Args:
            model_type: "groq" or "ollama"
            model_name: Model name (groq models or ollama models)
        """
        self.model_type = model_type
        self.embeddings = FastEmbedEmbeddings()
        
        # Initialize LLM based on type
        if model_type.lower() == "groq":
            self.llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=model_name,
                temperature=0.2,
            )
            print(f"✅ Initialized Groq LLM: {model_name}")
        else:  # ollama
            self.llm = Ollama(
                model=model_name,
                temperature=0.2
            )
            print(f"✅ Initialized Ollama LLM: {model_name}")
        
        # Initialize BERT topic extractor (lazy loading)
        self.bert_extractor = None
        
        self.vector_store = None
        self.transcript_data = []
        self.transcript_text = ""
        self.retriever = None
        self.chapters = []
        self.video_id = None
        self.bert_topics = None
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats"""
        parsed_url = urlparse(url)
        
        if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
            return parsed_url.path[1:]
        
        if parsed_url.hostname in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
            if parsed_url.path == '/watch':
                query_params = parse_qs(parsed_url.query)
                return query_params.get('v', [None])[0]
            
            if parsed_url.path.startswith(('/embed/', '/v/')):
                return parsed_url.path.split('/')[2]
        
        return None
    
    def fetch_transcript(self, video_url: str) -> Dict:
        """Fetch transcript with timestamps - ORIGINAL METHOD RESTORED"""
        video_id = self.extract_video_id(video_url)
        self.video_id = video_id
        
        if not video_id:
            return {"success": False, "error": "Invalid YouTube URL"}
        
        try:
            ytt_api = YouTubeTranscriptApi()
            self.transcript_data = ytt_api.fetch(video_id)
            self.transcript_text = " ".join(chunk.text for chunk in self.transcript_data)
            
            return {
                "success": True,
                "video_id": video_id,
                "transcript_length": len(self.transcript_text),
                "duration": self.transcript_data[-1].start if self.transcript_data else 0
            }
            
        except TranscriptsDisabled:
            return {"success": False, "error": "No captions available for this video"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def create_chunks_with_timestamps(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
        """Create chunks while preserving timestamp information"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Create chunks from text
        text_chunks = splitter.create_documents([self.transcript_text])
        
        # Map chunks back to approximate timestamps
        chunks_with_timestamps = []
        current_pos = 0
        
        for chunk in text_chunks:
            chunk_text = chunk.page_content
            
            # Find approximate timestamp for this chunk
            timestamp = self._find_timestamp_for_position(current_pos)
            
            chunks_with_timestamps.append({
                "text": chunk_text,
                "timestamp": timestamp,
                "formatted_time": self._format_timestamp(timestamp)
            })
            
            current_pos += len(chunk_text)
        
        return chunks_with_timestamps
    
    def _find_timestamp_for_position(self, char_position: int) -> float:
        """Find timestamp for a character position in the transcript"""
        accumulated_length = 0
        
        for entry in self.transcript_data:
            accumulated_length += len(entry.text) + 1  # +1 for space
            if accumulated_length >= char_position:
                return entry.start
        
        return self.transcript_data[-1].start if self.transcript_data else 0
    
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    
    def generate_youtube_link(self, timestamp: float) -> str:
        """Generate YouTube link with timestamp"""
        if not self.video_id:
            return ""
        seconds = int(timestamp)
        return f"https://www.youtube.com/watch?v={self.video_id}&t={seconds}s"
    
    def extract_topics_with_bert(self, num_clusters: int = 5) -> List[Dict]:
        """
        NEW METHOD: Extract topics using BERT embeddings and clustering
        This provides a more semantically-aware topic extraction
        """
        # Lazy load BERT
        if self.bert_extractor is None:
            self.bert_extractor = BERTTopicExtractor()
        
        # Create text segments for clustering
        segment_size = max(len(self.transcript_data) // 20, 10)
        segments = []
        
        for i in range(0, len(self.transcript_data), segment_size):
            window = self.transcript_data[i:i + segment_size]
            text = " ".join(chunk.text for chunk in window)
            segments.append(text[:500])  # Limit length
        
        # Run BERT clustering
        self.bert_topics = self.bert_extractor.cluster_topics(
            segments, 
            n_clusters=num_clusters
        )
        
        # Generate topic descriptions using LLM
        topics_with_descriptions = []
        
        for cluster_id, items in self.bert_topics['clusters'].items():
            # Get representative text from cluster
            sample_texts = [item['text'] for item in items[:3]]
            combined_text = " ".join(sample_texts)
            
            # Generate topic description using LLM
            prompt = f"""Analyze this cluster of related transcript segments and provide:
            1. A concise topic name (2-4 words)
            2. A brief description (1 sentence)
            
            Segments: {combined_text}
            
            Respond in JSON format: {{"topic": "...", "description": "..."}}"""
            
            try:
                response = self.llm.invoke(prompt)
                # Extract JSON from response
                if hasattr(response, 'content'):
                    response_text = response.content
                else:
                    response_text = str(response)
                
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    topic_data = json.loads(json_match.group())
                    topic_data['cluster_size'] = len(items)
                    topic_data['importance'] = "High" if len(items) > len(segments) / num_clusters else "Medium"
                    topics_with_descriptions.append(topic_data)
            except Exception as e:
                print(f"Error processing cluster {cluster_id}: {e}")
        
        print(f"✅ Extracted {len(topics_with_descriptions)} BERT-based topics")
        return topics_with_descriptions
    
    def detect_topic_changes(self, window_size: int = 30, similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Improved topic-boundary detection using:
        - time-based segments
        - semantic drift detection (centroid comparison)
        """

        if not self.transcript_data:
            return []

        print("🔍 Creating time-based segments...")

        MAX_DURATION = window_size
        segments = []
        current_segment = []
        current_start_time = self.transcript_data[0].start

        # --- TIME-BASED SEGMENTATION ---
        for entry in self.transcript_data:
            current_segment.append(entry)

            if entry.start - current_start_time >= MAX_DURATION:
                text = " ".join(c.text for c in current_segment)
                segments.append({
                    "text": text,
                    "timestamp": current_start_time
                })
                current_segment = []
                current_start_time = entry.start

        if current_segment:
            text = " ".join(c.text for c in current_segment)
            segments.append({
                "text": text,
                "timestamp": current_start_time
            })

        print(f"📦 Created {len(segments)} segments")

        # --- EMBEDDINGS ---
        print("🔧 Computing embeddings...")
        embeddings = [self.embeddings.embed_query(seg["text"]) for seg in segments]

        # --- TOPIC DRIFT DETECTION ---
        print("🎯 Detecting true topic boundaries...")

        topic_boundaries = [0]  # always begin at start
        topic_centroid = embeddings[0]  # semantic center of current topic

        for i in range(1, len(embeddings)):
            prev_sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            centroid_sim = cosine_similarity([topic_centroid], [embeddings[i]])[0][0]

            print(f"Segment {i} — prev_sim={prev_sim:.3f}, centroid_sim={centroid_sim:.3f}")

            # TOPIC CHANGE OCCURS IF BOTH CONDITIONS FAIL
            if (
                prev_sim < similarity_threshold
                and centroid_sim < (similarity_threshold + 0.05)
            ):
                topic_boundaries.append(i)
                topic_centroid = embeddings[i]  # reset topic center on change

        # Always include last segment
        if topic_boundaries[-1] != len(segments) - 1:
            topic_boundaries.append(len(segments) - 1)

        # Build final result
        result_segments = []
        for idx in topic_boundaries:
            seg = segments[idx]
            seg["index"] = idx
            result_segments.append(seg)

        print(f"✅ Final chapters detected: {len(result_segments)}")
        return result_segments


    
    def generate_chapter_titles(self, topic_segments: List[Dict]) -> List[Dict]:
        """Generate descriptive chapter titles using LLM"""
        chapters = []
        
        print(f"📝 Generating titles for {len(topic_segments)} chapters...")
        
        prompt_template = PromptTemplate(
            template="""Based on this transcript segment, create a concise, descriptive chapter title (3-6 words).
            The title should capture the main topic or theme discussed.
            
            Transcript segment:
            {text}
            
            Return ONLY the chapter title, nothing else.""",
            input_variables=["text"]
        )
        
        chain = prompt_template | self.llm | StrOutputParser()
        
        for i, segment in enumerate(topic_segments):
            # Limit text length for LLM processing
            text_preview = segment["text"][:500]
            
            title = chain.invoke({"text": text_preview}).strip()
            
            # Clean up the title
            title = title.strip('"').strip("'").strip()
            if title.lower().startswith("chapter"):
                title = title.split(":", 1)[-1].strip()
            
            chapter = {
                "index": i + 1,
                "title": title,
                "timestamp": segment["timestamp"],
                "formatted_time": self._format_timestamp(segment["timestamp"]),
                "youtube_link": self.generate_youtube_link(segment["timestamp"])
            }
            
            chapters.append(chapter)
            print(f"  ✓ Chapter {i+1}: {title} @ {chapter['formatted_time']}")
        
        return chapters
    
    def generate_chapters(self, window_size: int = 30, similarity_threshold: float = 0.7) -> List[Dict]:
        """
        Main method to detect topic changes and generate chapters
        """
        print("\n🎬 Starting automatic chapter generation...")
        
        # Detect topic changes
        topic_segments = self.detect_topic_changes(
            window_size=window_size,
            similarity_threshold=similarity_threshold
        )
        
        if not topic_segments:
            print("⚠️ Could not detect topic changes")
            return []
        
        print(f"✓ Detected {len(topic_segments)} topic boundaries")
        
        # Generate chapter titles
        self.chapters = self.generate_chapter_titles(topic_segments)
        
        print(f"\n✅ Successfully generated {len(self.chapters)} chapters")
        return self.chapters
    
    def build_vector_store(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Build FAISS vector store with timestamp metadata"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.create_documents([self.transcript_text])
        
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        return {"success": True, "num_chunks": len(chunks)}
    
    def generate_summary(self, style: str = "executive") -> str:
        """Generate video summary in different styles"""
        styles = {
            "executive": """Create a concise executive summary (3-4 sentences) 
                          highlighting the key points and main takeaways.""",
            "detailed": """Create a detailed summary covering all major topics, 
                         concepts, and important details discussed.""",
            "bullet": """Create a bullet-point summary with:
                       - Main topics (3-5 points)
                       - Key insights (3-5 points)
                       - Important conclusions"""
        }
        
        prompt_template = PromptTemplate(
            template="""Based on this video transcript, {style_instruction}
            
            Transcript:
            {transcript}
            
            Summary:""",
            input_variables=["transcript", "style_instruction"]
        )
        
        style_instruction = styles.get(style, styles["executive"])
        
        chain = prompt_template | self.llm | StrOutputParser()
        summary = chain.invoke({
            "transcript": self.transcript_text[:4000],  # Limit for context
            "style_instruction": style_instruction
        })
        
        return summary
    
    def extract_topics(self, num_topics: int = 5) -> List[Dict]:
        """Extract main topics and their importance"""
        prompt = PromptTemplate(
            template="""Analyze this transcript and extract the {num_topics} main topics discussed.
            
            For each topic, provide:
            1. Topic name (short, 2-4 words)
            2. Brief description (1 sentence)
            3. Importance (High/Medium/Low)
            
            Return as JSON array with format: [{{"topic": "...", "description": "...", "importance": "..."}}]
            
            Transcript:
            {transcript}
            
            Topics (JSON only):""",
            input_variables=["transcript", "num_topics"]
        )
        
        chain = prompt | self.llm | StrOutputParser()
        result = chain.invoke({
            "transcript": self.transcript_text[:4000],
            "num_topics": num_topics
        })
        
        # Parse JSON response
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                topics = json.loads(json_match.group())
                return topics
        except:
            pass
        
        return [{"topic": "Analysis failed", "description": result[:200], "importance": "Unknown"}]
    
    def create_qa_chain(self):
        """Create the RAG Q&A chain"""
        prompt = PromptTemplate(
            template="""You are a helpful assistant analyzing a YouTube video.
            Answer ONLY from the provided transcript context.
            If the context is insufficient, say you don't know.
            
            When relevant, mention approximate timestamps where information was discussed.

            If user asks something related to the topics in the video but that thing is not mentioned in video, mention it and provide user with an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer:""",
            input_variables=['context', 'question']
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        parallel_chain = RunnableParallel({
            'context': self.retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        
        main_chain = parallel_chain | prompt | self.llm | StrOutputParser()
        
        return main_chain
    
    def ask_question(self, question: str) -> str:
        """Ask a question about the video"""
        if not self.retriever:
            return "Error: Vector store not initialized. Please process a video first."
        
        chain = self.create_qa_chain()
        answer = chain.invoke(question)
        
        return answer
    
    def analyze_video(self, video_url: str, generate_chapters: bool = True, use_bert_topics: bool = False) -> Dict:
        """Complete video analysis pipeline"""
        print("📥 Fetching transcript...")
        transcript_result = self.fetch_transcript(video_url)
        
        if not transcript_result["success"]:
            return transcript_result
        
        print("✅ Transcript fetched successfully")
        print(f"📊 Video duration: {transcript_result['duration']:.0f} seconds")
        print(f"📄 Transcript length: {transcript_result['transcript_length']} characters")
        
        print("\n🔨 Building vector store...")
        vector_result = self.build_vector_store()
        print(f"✅ Created {vector_result['num_chunks']} chunks")
        
        print("\n📋 Generating summary...")
        summary = self.generate_summary(style="executive")
        
        # Topic extraction - use BERT if requested
        if use_bert_topics:
            print("\n🏷️ Extracting topics using BERT...")
            topics = self.extract_topics_with_bert()
        else:
            print("\n🏷️ Extracting topics...")
            topics = self.extract_topics()
        
        chapters = []
        if generate_chapters:
            chapters = self.generate_chapters()
        
        return {
            "success": True,
            "video_id": transcript_result["video_id"],
            "summary": summary,
            "topics": topics,
            "chapters": chapters,
            "ready_for_qa": True,
            "model_used": f"{self.model_type}",
            "bert_topics_used": use_bert_topics
        }