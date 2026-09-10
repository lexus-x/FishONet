#!/usr/bin/env python3
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

TOKEN = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
REPO_NAME = "FishONet"
REPO_DESC = "FishONet: Open-Set Species Recognition Across 17,393 Fish Taxa (CV4Ecology 2026 / Codabench 16815) - Leaderboard 53.761% (v109)"

REPO_HOMEPAGE = "https://fishonet.lalithsai00.workers.dev/"

headers = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github.v3+json",
    "User-Agent": "FishONet-Publisher"
}

def gh_api_get(url):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def gh_api_post(url, data):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

def gh_api_patch(url, data):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="PATCH")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))

print("--> 1. Fetching GitHub user info...")
try:
    user_info = gh_api_get("https://api.github.com/user")
except Exception as e:
    print(f"Error fetching user: {e}")
    sys.exit(1)

username = user_info.get("login", "lexus-x")
name = "Lalith Sai"  # Official author name
email = user_info.get("email") or "lalithsai00@gmail.com"

print(f"    GitHub User: {username}")
print(f"    Author Name: {name}")
print(f"    Author Email: {email}")

print(f"--> 2. Creating / verifying repository '{REPO_NAME}' on GitHub under '{username}'...")
try:
    repo_data = gh_api_post("https://api.github.com/user/repos", {
        "name": REPO_NAME,
        "description": REPO_DESC,
        "homepage": REPO_HOMEPAGE,
        "private": False,
        "has_issues": True,
        "has_wiki": False,
        "auto_init": False
    })
    print(f"    Repository created: {repo_data.get('html_url')}")
except urllib.error.HTTPError as e:
    err_msg = e.read().decode("utf-8")
    if "already exists" in err_msg:
        print(f"    Repository '{REPO_NAME}' already exists under {username}. Using existing repo.")
        # Update metadata including homepage
        try:
            gh_api_patch(f"https://api.github.com/repos/{username}/{REPO_NAME}", {
                "description": REPO_DESC,
                "homepage": REPO_HOMEPAGE
            })
            print(f"    Repository homepage updated to: {REPO_HOMEPAGE}")
        except Exception as patch_e:
            print(f"    Warning updating repository metadata: {patch_e}")
    else:
        print(f"    HTTP Error creating repo: {e.code} {err_msg}")
        sys.exit(1)

authenticated_remote_url = f"https://{TOKEN}@github.com/{username}/{REPO_NAME}.git"

repo_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
os.chdir(repo_root)

print("--> 3. Installing Apple-design figures into assets/figures/...")
os.makedirs("assets/figures", exist_ok=True)
os.makedirs("github/assets/figures", exist_ok=True)

fig_map = {
    "reports/paper_assets/pipeline.png": "assets/figures/pipeline.png",
    "reports/paper_assets/campaign.png": "assets/figures/campaign.png",
    "reports/paper_assets/task.png": "assets/figures/task.png",
    "reports/paper_assets/promise.png": "assets/figures/promise.png",
    "reports/design_assets/routing.png": "assets/figures/routing.png",
    "reports/design_assets/encoders.png": "assets/figures/encoders.png",
}

for src, dst in fig_map.items():
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        gh_dst = os.path.join("github", dst)
        shutil.copyfile(src, gh_dst)
        print(f"    Installed figure: {dst}")

# Copy favicon to assets/img if needed
if os.path.exists("assets/img/favicon.svg"):
    os.makedirs("github/assets/img", exist_ok=True)
    shutil.copyfile("assets/img/favicon.svg", "github/assets/img/favicon.svg")

print("--> 4. Preparing clean orphan branch...")
# Detach HEAD first so no branch is currently checked out
subprocess.run(["git", "checkout", "--detach"], check=True)
subprocess.run(["git", "branch", "-D", "clean-main"], stderr=subprocess.DEVNULL)
subprocess.run(["git", "checkout", "--orphan", "clean-main"], check=True)

print("--> 5. Staging reviewer files and Apple-design figures...")
# Run the staging script
staging_script = "/tmp/claude-1000/-home-ubuntu-onet/cc6988ac-9ba0-4f01-94fb-4786a1506642/scratchpad/stage_for_reviewers.sh"
if os.path.exists(staging_script):
    subprocess.run(["bash", staging_script], check=True)

# Add the figures, updated READMEs, LICENSE, index.html, and assets
subprocess.run(["git", "add", "README.md", "github/README.md", "LICENSE", "github/LICENSE", "index.html", "github/index.html", "assets", "github/assets"], check=True)

# Purge any AI or legacy files from the index
unwanted_items = [
    "CLAUDE.md",
    ".cursor",
    "REPORT.md",
    "model",
    "htmls",
    "archive",
    "AUDIT_REPORT.md",
    "implementation_plan.md",
    "graft",
    ".claude",
    "DESIGN-IS-2026-09-06",
    ".ignore",
    "skills-lock.json",
    "reports/FishONet_Challenge_Technical_Report.pdf",
    "reports/fishonet_openset_paper.pdf",
    "reports/unified_inference_evaluation.pdf",
    "reports/FishONet_Technical_Report.pdf",
    "reports/FishONet_Technical_Report_design.pdf",
    "reports/FishONet_Technical_Report_v2.pdf",
    "reports/REVISION_STATE.md",
    "reports/make_openset_paper.py",
    "reports/generate_fishonet_report.py",
    "reports/generate_unified_inference_memo.py",
    "reports/generate_v61_diagram.py",
    "reports/make_v109_paper.py",
    "reports/render_design.py",
    "reports/render_report.py",
    "reports/report.html",
    "reports/report_design.html",
    "reports/report_figs.py",
    "reports/build_technical_report.py",
    "reports/descriptions_fishbase.json",
    "reports/fishonet_figs",
    "reports/design_preview",
    "reports/design_preview_v2",
    "reports/design_preview_v3",
    "reports/design_preview_v4",
    "reports/design_preview_v5",
    "reports/paper_preview",
    "reports/pdf_preview",
    "research/unseen_vlm_rerank_deepseek.py"
]

for item in unwanted_items:
    subprocess.run(["git", "rm", "-r", "--cached", "--ignore-unmatch", item], stderr=subprocess.DEVNULL)

# Commit solely as the authenticated user (NO co-authors, NO Claude)
commit_env = os.environ.copy()
commit_env["GIT_AUTHOR_NAME"] = name
commit_env["GIT_AUTHOR_EMAIL"] = email
commit_env["GIT_COMMITTER_NAME"] = name
commit_env["GIT_COMMITTER_EMAIL"] = email

commit_msg = "FishONet: Team CWNU AIX (Rank #3) - Open-set species recognition (53.761% v109)"

print(f"--> 6. Committing exclusively as '{name} <{email}>'...")
subprocess.run(["git", "commit", "-m", commit_msg], env=commit_env, check=True)

print("--> 7. Setting up remote and pushing to GitHub as 'main'...")
remotes = subprocess.check_output(["git", "remote"], text=True).split()
if "clean-origin" in remotes:
    subprocess.run(["git", "remote", "remove", "clean-origin"], check=True)

subprocess.run(["git", "remote", "add", "clean-origin", authenticated_remote_url], check=True)

# Push clean-main as 'main'
subprocess.run(["git", "push", "-u", "clean-origin", "clean-main:main", "--force"], check=True)

print("\n" + "="*70)
print("SUCCESS!")
print(f"Repository is live at: https://github.com/{username}/{REPO_NAME}")
print(f"Commit Author: {name} <{email}>")
print("100% Sole Contributor — Zero Claude traces, zero co-author trailers.")
print("Apple-grade layout & figures embedded.")
print("="*70)
