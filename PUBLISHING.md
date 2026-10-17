# Publishing the code and project page

This folder contains the Python source, processed EHG data, figures, numerical results, and the static project page. The raw WFDB source archive is supplied separately and can also be obtained from [PhysioNet](https://physionet.org/content/tpehgdb/1.0.1/).

## GitHub repository

Create an empty repository, then run the following from this folder:

```bash
git init
git add .
git commit -m "Release EHG preterm birth analysis code"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

## Project page

Go to **Settings → Pages** in the repository, choose **Deploy from a branch**, select `main`, and set the folder to `/docs`. The interactive project page is located at `docs/index.html`. All its figures and metric data are bundled locally.

To regenerate the page after an experiment, run:

```bash
python scripts/build_site.py
```

## Dataset and licenses

The data guide is at `docs/DATA.md`; source-license information is retained in `docs/UPSTREAM_EXPERIMENTS_LICENSE.txt`. Cite the paper and PhysioNet TPEHGDB when using the project.
