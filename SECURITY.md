# Security Policy

The Crate is a **local-first** application: it runs on your own machine and your audio,
library, sessions and AI prompts never leave it. The only outbound traffic is the optional,
opt-in Discogs / Resident Advisor / web-search lookups, which send only the search terms
for the request you make. This narrows the attack surface a lot, but a few things still
matter — please read below.

## Reporting a vulnerability

Please report security issues **privately**, not in a public issue:

- Preferred: open a [GitHub private security advisory](https://docs.github.com/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
  on this repository (**Security → Report a vulnerability**).
- Or email the maintainer (see the commit history / GitHub profile).

Include steps to reproduce and the impact you observed. You'll get an acknowledgement as
soon as possible; please allow time for a fix before any public disclosure.

## Secrets & credentials

- **Never commit `.env`.** It is git-ignored; only `.env.example` (placeholders) is tracked.
  All credentials come from the environment — none are hard-coded.
- The only required secret is `POSTGRES_PASSWORD`. `DISCOGS_ACCESS_TOKEN` is optional
  (enrichment only).
- If you cloned an early version or shared your machine, **rotate** any real Discogs token
  and set a strong, unique `POSTGRES_PASSWORD`.
- The database listens on `127.0.0.1` only by default. Do not expose Postgres or the API
  to a public network without putting authentication in front of them.

## Scope notes

- The AI assistant can fetch web content (Resident Advisor, your registered reference
  sites). Treat retrieved text as untrusted data; the assistant is instructed to ground
  answers in tool results and not to act on retrieved instructions.
- Audio import standardises and re-encodes files; filenames are sanitised before they
  touch disk.

## Supported versions

This is an actively developed project; fixes land on `main`. There is no separate LTS
branch — please test against the latest `main`.
