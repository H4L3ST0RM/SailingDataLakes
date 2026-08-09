# SailingDataLakes

Source for [sailingdatalakes.com](https://sailingdatalakes.com) — machine
learning and applied math walkthroughs, written from first principles
(concept, math, and from-scratch implementation), plus side projects that
apply ML to things like sailing and baseball.

Each post/project starts as a Jupyter notebook, which gets converted to a
Hugo markdown page via `nbconvert`. The notebook is kept alongside the
generated page so the code is fully reproducible.

## Stack

- [Hugo](https://gohugo.io/) static site generator
- [hugo-coder](https://github.com/luizdepra/hugo-coder) theme (vendored as
  a git submodule under `themes/hugo-coder`)
- Content authored in Jupyter, converted with `nbconvert`
- Deployed to GitHub Pages via GitHub Actions on every push to `main`
  (see [`.github/workflows/hugo.yml`](.github/workflows/hugo.yml))

## Structure

```
content/
  posts/       ML/math walkthrough posts, one directory per post
  projects/    Side projects, one directory per project
  about.md
  resume.md
static/        Static assets (images, robots.txt)
assets/scss/   Site-specific style overrides
layouts/       Site-specific layout/partial overrides (theme is never
               edited directly — see below)
themes/hugo-coder/   Vendored theme, git submodule
```

Each post/project directory contains:
- the source `.ipynb` notebook (front matter lives in a raw cell at the top)
- the generated `index.md` (what Hugo actually renders)
- any plot images nbconvert produced, plus hand-supplied illustrations
  where relevant

## Running locally

```bash
git clone --recurse-submodules https://github.com/H4L3ST0RM/SailingDataLakes.git
cd SailingDataLakes
hugo server -D
```

`-D` includes draft content. The site is served at `http://localhost:1313`.

## Notes for contributors (i.e. future me)

- The `hugo-coder` theme is a **git submodule** — never edit files under
  `themes/hugo-coder/` directly, those changes won't be tracked. Site
  styling goes in `assets/scss/custom.scss`; layout overrides go in this
  repo's own `layouts/` (Hugo prefers the project's layouts over the
  theme's for any matching path).
- Merging to `main` **is** the deploy — there's no separate release step.
