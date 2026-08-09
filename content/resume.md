+++
title = "Resume"
description = "My Work Experience"
date = "2014-03-12"
lastmod = "2026-08-09"
aliases = ["resume", "cv", "work-experience"]
author = "John C Hale"
draft = true
+++
John C. Hale
============

**Data-Driven Fraud Prevention**

-------------------     ----------------------------
Boerne, TX | john.c.hale@proton.me | [LinkedIn](https://www.linkedin.com/in/john-c-hale/)
-------------------     ----------------------------

Technical Proficiencies
------------------------
**Data Science**
:   Python, SQL, Github, Shell, C++, Object-Oriented Programming, Functional Programming,
    Databricks, Data Analysis, Data Visualization, PySpark, Jupyter Notebook, Statistics,
    Tableau, Communication

**Machine Learning**
:   Feature Engineering, Feature Reduction, Scikit-Learn, Supervised Learning, Classification,
    Regression, Unsupervised Learning, Cluster Analysis, Imbalanced Datasets, Anomaly Detection,
    Data Analysis

Experience
----------

**Senior Fraud Strategy Manager at Imprint Payments** | 2026 Apr - Present

Lead fraud strategy for first party fraud (FPF). Heavily utilize AI (Claude Code), data
expertise, Github, Notion, Sardine AI, Alloy, and Linear.

*Leadership & Training*

* Developed the definition for first party fraud, along with a KPI and a loss budget.
* Developed an FPF framework to derive target KPIs based on the holistic economic impact of a treatment at a given control point.
* Managed relationships with various vendors and partner banks.

*Technical*

* Piloted and launched a geo-monitoring rule at onboarding with an expected 20% recall and 10% precision.
* Developed various transaction-level rules targeting FPF/credit abuse behavior and deployed them via Sardine.
* Researched and deployed various payment-level rules targeting payment abuse behavior.

**Data Scientist at Sardine AI** | 2025 Jul - 2026 Apr

Led client Proof of Concept (POC) engagements, ensuring Sardine's product suite met customer needs during trial periods. Heavily utilized GCP, primarily BigQuery and Google Vertex.

* Managed multiple concurrent POCs, including potential six-figure deals, through regular client meetings, deep data analysis, and fraud trend discovery to demonstrate product value.
* Evaluated third-party data providers in ambiguous problem spaces, designing creative evaluation frameworks.
* Developed custom fraud models tailored to client use cases as part of POC delivery.

**Senior Data Scientist at Acorns** | 2022 Feb - 2025 Jul

Specialized in collaborating with the risk organization, providing machine learning solutions as well as guidance on fraud preventive measures. Acorns data science heavily utilized Databricks on AWS.

*Technical*

* Added multiple sub-models to a customer-facing recommendation engine, increasing engagement by ~15% over the previous model.
* Initiated the referral fraud program, advocating for its need and creating the first set of preventive rules; built two unsupervised models for bot detection using UMAP and HDBSCAN. This program has saved the company around $6 million over 2 years.
* Developed and maintained an internal Python package for fraud strategy generation and threshold optimization, letting users quickly generate rulesets and optimize them directly to a business objective.
* Used XGBoost to build a new account fraud model, deployed for batch inferencing through Databricks. Suspends several hundred fraudulent new accounts per month.
* Created a deposit fraud model using XGBoost, deployed via Databricks for batch inference. Prevents several hundred fraudulent transactions per month, with an estimated $12k/month saved.
* Built out Tableau dashboards for monitoring and tracking metrics across fraud channels and controls.

*Leadership & Training*

* Started and led an initiative to define the ideal-state fraud prevention program at Acorns, including fraud channels, controls, metrics, and a roadmap to get there.
* Led a cross-functional monthly forum for fraud and AML prevention, typically attended by 20 people including C-suite executives.
* Started and led a PoC using contextual multi-armed bandit testing for referral communication optimization.

**Data Scientist at Charles Schwab** | 2019 Aug - 2022 Feb

First hire on the data science team embedded within the financial crimes organization, working closely with fraud strategists and analysts to identify and implement advanced solutions. This team primarily worked on Schwab's on-prem Hadoop cluster (Jupyter Notebooks), Teradata, and MySQL.

*Technical*

* Created an ensemble model (logistic regression, XGBoost, and an MLP) for detecting client impersonation in the call center with ~15% precision, demonstrated to prevent several million in fraud losses annually.
* Developed a login segmentation for risk scoring using K-Means and prototyped deploying it via a Flask API; demonstrated it could increase precision in risk transaction rules by roughly 5% with no impact on recall.
* Worked as part of a large cross-functional team to build a solution for detecting victims of microcap pump-and-dump schemes, using time series analysis and unsupervised learning; maintained the deployment.
* Built an auto-regressive time series model for anomaly detection in applicant volumes, automating reporting and alerting the risk team to potential fraud attacks.

*Leadership & Training*

* Led a monthly risk forum on relevant projects and fraud trends, regularly attended by 30+ people, ranging from ICs to managing directors.
* Pitched an initiative to reassess organizational metrics and add dashboarding and monitoring; the pitch was successful and a 4-person cross-functional task force completed the ~6-month project.
* Initiated and led a weekly Python coding bootcamp for fraud strategists and analysts (10 participants), covering programming basics, algorithms, and data analysis/visualization with Pandas, NumPy, Matplotlib, and Seaborn.

**DevOps Engineer at Charles Schwab** | 2017 Jul - 2019 Aug

Consulted teams across the tech organization to spread adoption of best practices and tooling, searching for and implementing automation opportunities.

* Implemented server/service restart automations for 11 different applications using Micro Focus Operation Orchestration (OO), saving an estimated 45 hrs/month of labor.
* Built two access provisioning automations and a generic automation for syncing distribution lists with databases using OO, saving an estimated 100 hrs/month of labor.
* Designed an extensive CI/CD process for a large application using Bamboo, coordinating across multiple developer and IT teams and taking the team from a 6-week build/deploy cycle to a weekly one.

Education
---------
2020
:   **M.S. Data Science** | Texas Tech University

2017
:   **B.S. Computer Science** | Texas Tech University
:   **B.S. Mathematics** | Texas Tech University

Personal Projects
------------------
Sailing Data Lakes Blog | 2024 - Present
:   Used the Hugo static site generator, along with GitHub Actions, to build and deploy content
    to [my website](https://sailingdatalakes.com/).

Sailboat Tack Optimizer
:   Built a Q-learning agent that learns to sail a simulated boat to a
    waypoint, optimizing tack decisions against wind direction and
    velocity made good. Modeled the sailing environment (wind, points of
    sail, tack maneuvers) and reward shaping from scratch in Python.
    See the [full write-up](https://sailingdatalakes.com/projects/sailing-route-optimization/).
