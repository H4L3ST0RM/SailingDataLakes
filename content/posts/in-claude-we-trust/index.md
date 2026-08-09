+++
authors = ["John C Hale"]
title = "In Claude We Trust? Accountability and Traceability in AI-Assisted Fraud Analysis"
date = "2026-08-09"
description = "Why AI-assisted fraud analysis needs the same rigor as production code — reproducibility, traceability, and human checkpoints, not just faster output."
math = false
draft = false
tags = [
    "fraud",
    "risk",
    "ai",
    "data science",
    "reproducibility",
    "governance",
]
categories = [
    "Fraud & Risk Commentary",
]
+++

Everyone I work with uses Claude now — generating analyses, writing up docs, building the case for or against a decision. It's made us faster. It's also surfaced a set of problems that I think the industry, broadly, hasn't caught up to yet.

I've noticed two flavors of this across teams I've worked with. More experienced people tend to present AI-assisted analysis with enough confidence that nobody thinks to double-check it. Less experienced folks get the opposite treatment — they present, get a round of questions, and then have to scramble to reconstruct a result that isn't easy to recreate. I started using git for my own work from day one, so I've mostly sidestepped this myself — but it's a pattern worth naming, not a complaint about any one team.

The trust question comes up often, too. It's routine to hear "this was generated with Claude" tacked onto a report, and in practice that caveat rarely triggers a deeper conversation about how the analysis was actually produced. Everyone's encouraged to use these tools — as they should be — but "we all use it" isn't the same thing as "we can all verify it." In Claude we trust, right?

Onboarding surfaces the same gap from a different angle. My first several weeks anywhere new tend to involve a lot of searching and asking around, trying to understand what tables people are using, where the data comes from, what assumptions are baked in. Usually I get pointed to a writeup, which helps, and sometimes even includes a high-level methodology — but it rarely gets down to the actual tables and queries. Knowing the underlying sources of truth from day one saves everyone time.

And it's not just an individual habit — it can become a real coordination problem. I've seen two teams whose metrics *should* align (same loss category, different reporting lineages) end up with numbers that don't quite match, purely because the definitions and calculations behind them were never documented against a shared standard. Nobody did anything wrong — the mismatch was just the natural result of two teams building independently without a shared, written source of truth to check against.

None of this is an argument against using AI for analysis. It's an argument that reproducibility, trust, and documentation don't come for free — you have to build them in deliberately, or they don't happen at all.

## What I actually do about it

I don't open a chat and start prompting. I start with a written problem statement — what happened, over what time period, with whatever I already know — filed as a ticket, not typed loose into a conversation.

From there, I run a structured process with five stages, each requiring my sign-off before moving to the next:

1. **Scoping.** Pin down the actual problem, the time period, and where in the funnel we're acting — before any data gets touched.
2. **Exploratory analysis.** Pull and validate the relevant data, and confirm the signals worth pursuing.
3. **Candidate generation.** Generate and evaluate candidate rules or approaches, and surface the strongest performers for review.
4. **Tuning.** Refine the chosen approach against the actual objective (precision, recall, dollar impact — whatever the ticket specified), then check back in on the final recommendation.
5. **Writeup and handoff.** Document the conclusion and open it for review — nothing gets finalized without a human reading it first.

Every analysis is version-controlled and filed under a naming convention tied back to the originating ticket, so anyone can trace a given number back to exactly what produced it, months later, without asking around. And nothing goes to production off the analysis alone — it gets a shadow-run validation against real-world results and a documented handoff before it ever goes live.

The specifics of how I've built this out are mine to keep close, but the shape of it isn't proprietary to me or my employer — it's just good practice most AI-assisted analysis today skips: write the problem down before you start, checkpoint with a human at each real decision, version-control the whole trail, and validate in the real world before you trust the conclusion.

None of this guarantees the underlying analysis is correct — a fully documented, git-tracked pipeline can encode a wrong assumption or a subtle bug just as easily as an undocumented one. What it guarantees is that the assumption or the bug is inspectable, by someone other than the person who made it, instead of disappearing into a chat window nobody else can open. Traceability doesn't replace scrutiny. It just makes scrutiny possible.

## The point

This doesn't fully close the gap I opened with, but it does two concrete things for someone earlier in their career than me. First, it gives them something to learn from — a consistent, documented framework instead of a free-for-all where every analyst invents their own process from scratch. Second, if they get asked a hard question about their own analysis later, they're not relying on memory — they can go back to the actual tracked data and code and work through it, prompting against the real artifact instead of trying to reconstruct what they did.

What it doesn't do, at least not yet, is fix cross-team alignment. The mismatch I described earlier between two teams' numbers isn't solved by my own discipline — it only gets solved if the other team adopts something similar too. If a risk org took this on holistically, I could see it going one of two ways: either each team's tracked analyses become something a shared agent can reference across teams, or the core shared metrics get generated by one common process everyone draws from, so consistency is built in rather than reconciled after the fact. Neither of those exists today. This is a personal practice, not an organizational one, and I think it's worth being honest about that.

If you can't reproduce a colleague's analysis, can't explain how a number was derived, and can't tell a new hire where the ground truth lives, none of that gets fixed by using AI more carefully. It gets fixed by treating the analysis itself — not just the conclusion — as something worth version-controlling, documenting, and handing off. AI didn't create this problem. It just made it a lot easier to skip the part where you write any of it down.
