[![Netlify Status](https://api.netlify.com/api/v1/badges/2f8f1ca8-27e3-4781-8493-aace97152622/deploy-status)](https://app.netlify.com/sites/hugo-graphite/deploys) [![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)

# hugo-graphite

This is a Hugo theme, forked from the Hugo Lithium theme by Jonathan Rutheiser and Yihui Xie, with modifications by Hadley Wickham, Mara Averick, Robby Shaver, and the tidyverse team, plus changes to make it work better for RStudio teams like [`tidyverse.org`](https://www.tidyverse.org/) by [Desirée De Leon](https://desiree.rbind.io) and [Alison Hill](https://alison.rbind.io). 

Why graphite? It's a mineral made up of carbon atoms that are arranged in hexes. Graphite is also in pencils-- which makes it familiar, easy to use, and easy to change (i.e. erasable). This theme also has several features that make it easy to use and change too:

+ **articles** (blog posts with a thumbnail image on the listing page and a banner image on the single page; all link with team member names or just plain text for guest contributors)
+ **authors** (team members who may contribute to articles; includes a team listing page plus individual bios that link to articles contributed to for each)
+ **events** (including upcoming event sticky reminders, and upcoming event calendar, and a past event archive organized by year; all link with team member names in attendance)

## Theme update 2020.07.01 

### New and improved features include:

* Gains a sticky Table of Contents for blog posts
* Gains new taxonomy terms templates for [tags](https://hugo-graphite.netlify.app/tags/) and [categories](https://hugo-graphite.netlify.app/categories/)
* Gains copy-to-clipboard functionality for code chunks
* Gains anchorized header links
* Gains new shortcodes to create note and warning boxes
* Gains two new layout types ([**tutorial**](https://hugo-graphite.netlify.app/start/) and [**learn**](https://hugo-graphite.netlify.app/learn/))
* Better image accessibility
* Larger base font sizes for improved accessibility
* Many more parameters for customizing the site aesthetics

### Small tweaks

* Adds `target=_blank` to all external links
* Hexes on package pages are now symmetric
* Improved functioning for highlighting active menu items in the upper navbar (even with Hugo [nested sections](https://gohugo.io/content-management/sections/#nested-sections))
* JQuery upgrade
* Improved RSS feed handling (see: https://hugo-graphite.netlify.app/blog/index.xml)
* Improved sharing images for opengraph and twitter
* Styled footnotes from Goldmark Renderer
* Includes chroma CSS styles for highlighting via downlit for Hugodown sites
* Shiny new demo site
* A new Hugo theme name that people can actually pronounce!


This theme is a *work in progress*. This means that the theme will be changing frequently as we rapidly iterate and explore variations to meet our needs. You are generally best off waiting until the theme is more mature before you use it. We'll update this `README` and the repo status when ready.
