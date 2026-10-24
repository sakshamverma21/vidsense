# Deploying VidSense to Azure Container Apps

**Plain-English version of what we're doing:** you already built a Docker image — a
sealed box containing the whole app. Azure Container Apps is a service that runs
that box and gives it a public `https://…` URL. It **scales to zero**, so it costs
≈nothing while nobody is using it. We install one tool (`az`), log in, and run a
handful of commands.

Run every command below in **PowerShell** (your normal terminal), from the project
folder unless told otherwise.

---

## 0. One-time: get free Azure credit (GitHub Student Pack)

1. Go to **https://azure.microsoft.com/free/students**
2. Sign in with the **student email** on your GitHub Student Pack.
3. Activate → you get **$100 credit, no credit card required**.

## 1. Install the Azure CLI (the `az` command)

```powershell
winget install -e --id Microsoft.AzureCLI
```
(If `winget` isn't available, download the installer: https://aka.ms/installazurecliwindows)

**Close and reopen your terminal**, then confirm it works:
```powershell
az version
```

## 2. Log in

```powershell
az login
```
A browser opens — pick your student account. If you have more than one subscription:
```powershell
az account list --output table
az account set --subscription "Azure for Students"
```

## 3. First-time setup (extension + enable the services)

```powershell
az extension add --name containerapp --upgrade
az provider register --namespace Microsoft.App
az provider register --namespace Microsoft.OperationalInsights
```

## 4. Create a resource group (a folder that holds everything)

```powershell
az group create --name vidsense-rg --location centralindia
```
(`centralindia` is close to you; `eastus` etc. also work.)

## 5. Prepare the starter collection and deploy

After a fresh Git checkout, install the runtime dependencies, configure `.env`,
and prepare the starter corpus locally before building the deployment image:

```powershell
python -c "from vidsense.corpus import load_corpus; load_corpus('vidsense/eval/corpus.json')"
```

Generated transcripts and FAISS indexes are intentionally excluded from Git.
This command may consume provider credits when no cache exists. Docker includes
only the prepared starter cache, not visitor-created collections. An image built
without this preparation still starts, but must fetch the starter transcripts at
runtime with a working transcript provider.

From the project folder (the one with the `Dockerfile`):
```powershell
az containerapp up --name vidsense --resource-group vidsense-rg --location centralindia --source . --ingress external --target-port 8000
```
This uploads your code, **builds the image in the cloud**, and deploys it (~5–10 min).
At the end it prints a URL like
`https://vidsense.<random>.centralindia.azurecontainerapps.io`.

For this repository's existing Azure app, omit `--location` so Azure updates the
current Korea Central environment instead of attempting to create another one:

```powershell
az containerapp up --name vidsense --resource-group vidsense-rg --source . --ingress external --target-port 8000
```

> The site loads with a prepared starter collection. Visitor collections now
> require MongoDB-backed accounts and verified email; configure MongoDB and SMTP
> using [the account deployment guide](docs/accounts-and-mongodb.md). Without
> these settings the deployment runs in demo-only mode. Research requires the
> model-provider keys in the next step.

## 6. Add your API keys (as secrets, not baked into the image)

Never put credentials in the Dockerfile, source code, image build arguments, or
public repository files. `.env` is excluded from both Git and Docker build
contexts. Inject secrets through the hosting platform's secret manager or
runtime environment variables. Before pushing, run:

```bash
python scripts/check_secrets.py
```

The repository CI workflows run tests plus Gitleaks, and the container runs as
a non-root user.

The current portfolio deployment uses Groq. Cloud-hosted source ingestion also
needs a managed transcript key because YouTube commonly blocks Azure data-center
IPs. Create a free starter key at https://supadata.ai/youtube-transcript-api.
Add the rotated values through the
Azure portal's **Container App → Secrets** page (recommended for beginners), or
use placeholders in the CLI command below. Do not commit or paste real values
into documentation.

```powershell
az containerapp secret set --name vidsense --resource-group vidsense-rg --secrets groq-key=YOUR_GROQ_KEY tavily-key=YOUR_TAVILY_KEY supadata-key=YOUR_SUPADATA_KEY

az containerapp update --name vidsense --resource-group vidsense-rg --set-env-vars VIDSENSE_PROVIDER=groq VIDSENSE_GROQ_MODEL=openai/gpt-oss-20b VIDSENSE_TRANSCRIPT_PROVIDER=auto GROQ_API_KEY=secretref:groq-key TAVILY_API_KEY=secretref:tavily-key SUPADATA_API_KEY=secretref:supadata-key
```
Replace the placeholders with the rotated values from your local `.env`.
The app restarts with the keys and demo research becomes available. Custom
imports also require the MongoDB and SMTP configuration described above.

To use a playlist size other than the default 50 videos, add
`VIDSENSE_MAX_PLAYLIST_VIDEOS=YOUR_LIMIT` to the same `--set-env-vars` command.
The default `VIDSENSE_EMBEDDING_BATCH_SIZE=16` is intentionally conservative
for small Azure containers; only raise it after measuring peak memory while
indexing a representative playlist.

### Keep the portfolio deployment inexpensive and session-safe

Custom collections are saved in MongoDB, while indexes and ingestion workers
remain local to one process. Allow scale-to-zero but keep one maximum replica
until ingestion is moved into a durable worker queue:

```powershell
az containerapp update --name vidsense --resource-group vidsense-rg --min-replicas 0 --max-replicas 1
```

The starter corpus survives scale-to-zero because it is baked into the image.
Visitor collections are restored from MongoDB; rebuilding the local index can
take time after a restart. Interrupted imports can be retried after their lease
expires, reusing the videos already saved to MongoDB.

## 7. Get your public URL

```powershell
az containerapp show --name vidsense --resource-group vidsense-rg --query properties.configuration.ingress.fqdn --output tsv
```
Open `https://` + that address. **That's your live app** — the link for your resume.

Current portfolio URL:
`https://vidsense.sakshamverma.dev`

## 8. Redeploy after code changes

Re-run the existing-app command from step 5; it rebuilds and rolls out a new
revision without changing regions.

## 9. ⚠️ Stop charges when you're done experimenting

Container Apps scales to zero, but the registry + logs cost a little. To remove
**everything** at once:
```powershell
az group delete --name vidsense-rg --yes --no-wait
```
You can always redeploy later. Use Azure Cost Management to create a monthly
budget and alerts. Container Apps can scale to zero, but registry storage, logs,
and other resources may still incur charges.

---

## Alternative: a free, always-on demo link (no Docker/Azure)

If you also want a link that's *always* up with zero cost/maintenance, **Hugging
Face Spaces** runs this same app for free:
1. Create a Space (SDK: **Docker**) at https://huggingface.co/new-space
2. Push this repo to it (it uses the same `Dockerfile`).
3. Add `VIDSENSE_PROVIDER=groq`, `GROQ_API_KEY`, `SUPADATA_API_KEY`, and
   optionally `TAVILY_API_KEY`
   in the Space's **Settings → Variables and secrets**.

Keep the Azure deploy for the "cloud deployment experience" on your resume, and the
Space as the durable demo link.
