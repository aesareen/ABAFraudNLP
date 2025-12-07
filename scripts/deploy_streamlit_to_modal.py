import shlex
import subprocess
from pathlib import Path
import modal
import os

# Define the image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .pip_install(
        "streamlit",
        "pandas",
        "altair",
        "python-dotenv",
        "supabase",
        "scikit-learn",
        "nltk",
        "pocket-agent>=0.2.23",
        "fastmcp>=2.13.0.2",
        "litellm>=1.79.0",
        "openai>=2.6.1",
        "rank-bm25>=0.2.2",
        "rich>=14.2.0",
        "torch>=2.9.1",
        "bert-extractive-summarizer>=0.10.1",
        "keybert>=0.9.0",
        "crawl4ai>=0.7.6",
        "vecs>=0.4.5",
        "polars>=1.33.1",
    )
    .run_commands("python -m nltk.downloader stopwords")
)

secrets_path = os.path.join(Path(__file__).parent.parent, "config", ".env")

app = modal.App(
    name="abafraudnlp",
    image=image,
    secrets=[modal.Secret.from_dotenv(path=secrets_path)],
)

# Mount the entire project directory to /root
image = image.add_local_dir(
    Path(__file__).parent.parent,
    remote_path="/root",
    # ignore these extraneous directories/files that would muddy our modal environment
    ignore=[
        "**/__pycache__",
        "**/.git",
        "**/.venv",
        "**/node_modules",
        "**/.history",
        "**/.cache",
        "**/.profile",
        "**/.git",
        "**/.claude/",
        "**/.gitignore",
        "**/.cache/",
        "**/.profile",
        "**/pocket-agent.log",
        "**/logs/",
    ],
)


@app.function(
    image=image,
    timeout=3600,
)
@modal.web_server(8000)
@modal.concurrent(max_inputs=100)
def run():
    # The Streamlit app is at /root/streamlit/streamlit_app.py in the container
    target = "/root/streamlit/streamlit_app.py"

    cmd = f"streamlit run {target} --server.port 8000 --server.enableCORS=false --server.enableXsrfProtection=false"

    # Streamlit needs to run in the root directory so that `import agents...` works
    subprocess.Popen(cmd, shell=True, cwd="/root")

    print("Streamlit app deployed to modal!")
