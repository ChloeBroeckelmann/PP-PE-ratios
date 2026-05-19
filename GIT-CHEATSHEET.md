# Git Sync Cheat Sheet

## After making changes — PUSH to GitHub

Open a terminal in this folder and run these 3 commands:

```
git add .
git commit -m "describe what you changed"
git push
```

Example commit message: `git commit -m "updated app.py calibration section"`

---

## When switching computers — PULL from GitHub

Run this one command:

```
git pull
```

---

## The workflow

1. Start work on Mac → `git pull` first
2. Make changes
3. `git add .` → `git commit -m "..."` → `git push`
4. Switch to Windows → `git pull`
5. Make changes
6. `git add .` → `git commit -m "..."` → `git push`
7. Switch back to Mac → `git pull`

**Always pull before you start, always push when you finish.**

---

## How to open a terminal in this folder (Windows)

- Open File Explorer and navigate to this folder
- Click the address bar at the top, type `cmd` and press Enter
- OR right-click inside the folder and choose "Open in Terminal"
