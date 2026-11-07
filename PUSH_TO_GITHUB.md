# Publish this project to the assigned GitHub repository

The source bundle is prepared for the empty repository:

`https://github.com/di0206-innovator/Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu`

From Terminal, run these commands after downloading the ZIP into your Downloads folder:

```bash
git clone https://github.com/di0206-innovator/Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu.git
cd Data-Type-Memory-Performance-Benchmark-Lab-Divyanshu
unzip ~/Downloads/DataType-Lab-GitHub-Project.zip -d .
git add .
git commit -m "Build DataType Lab benchmarking application"
git push -u origin main
```

If the ZIP is saved somewhere else, replace the ZIP path in the `unzip` command. GitHub may prompt you to authenticate when pushing.

After the push, refresh the repository page. The root should contain `app.py`, `benchmark_lab/`, `tests/`, `.github/`, the requirements files, and the project report.
