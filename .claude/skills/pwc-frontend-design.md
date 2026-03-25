---
name: pwc-frontend-design
description: Create frontend interfaces aligned with PwC 2026 brand standards. Use when building proposals, dashboards, presentations, or client-facing interfaces that must reflect PwC’s premium, consistent, and professional identity.
license: MIT
---

This skill generates frontend interfaces that strictly follow PwC brand principles while delivering high-quality, modern, and confident design.

The output should feel:
- Premium
- Clear
- Structured
- Confident
- Client-ready

NOT experimental, playful, or decorative.

---

# CORE BRAND PRINCIPLES (2026)

## 1. Clarity over decoration
- Design must communicate, not entertain
- Every element must have a purpose
- Remove anything that feels ornamental

## 2. Confidence and structure
- Strong hierarchy
- Clean layout
- Intentional spacing

## 3. Momentum and progress
- Subtle motion or flow
- Forward-moving layouts
- Visual direction (left → right, top → down)

## 4. AI-enabled future tone
- Clean, modern, but not “tech startup”
- Feels intelligent, not flashy
- Avoid gimmicky AI visuals

---

# VISUAL SYSTEM

## Colour

Primary:
- PwC Orange: #FD5108
- Black: #000000
- White: #FFFFFF

Rules:
- Orange is an accent, NOT a background
- Use sparingly for:
  - highlights
  - key numbers
  - CTAs

Avoid:
- Full orange backgrounds
- Random gradients
- Off-brand colours

---

## Typography

Use ONLY:

- Serif: Georgia
- Sans-serif: Arial

Rules:
- Headlines: Georgia
- Body: Arial
- No italics in headlines
- Text only in black or white

Avoid:
- Custom fonts
- Decorative fonts
- Over-styling text

---

## Layout

Structure:
- Grid-based
- Clean alignment
- Clear sections

Use:
- Generous whitespace
- Section separation
- Logical flow

Avoid:
- Overlapping elements
- Chaotic layouts
- Floating UI with no structure

---

## Components

### Panels / Sections
- Flat or very subtle elevation
- Minimal shadows
- Clear boundaries

### Buttons
- Simple rectangular or slightly rounded
- Orange for primary CTA
- Black/white for secondary

### Data Displays
- Clean charts
- Strong labels
- No unnecessary decoration

---

## Imagery

Use:
- Focus photography (hero, cover)
- Context photography (supporting)

Rules:
- Images only where meaningful
- Do not overuse imagery
- Never use decorative stock visuals

---

# LOGO USAGE (CRITICAL)

- Logo must appear on primary surfaces
- Do NOT:
  - distort
  - recolour
  - separate elements

Use asset:
`assets/pwc-logo.jpeg`

---

# ANTI-AI SLOP (STRICT)

Reject outputs that include:

- Generic SaaS dashboards
- Gradient-heavy UI
- “Modern startup landing page”
- Overuse of cards and shadows
- Purple/blue tech palettes
- Centre-aligned hero sections

If it looks like:
- a startup template
- a design system demo

→ FAIL

---

# CONTENT & TONE (IMPORTANT)

Follow PwC writing style:

- Use active voice
- Use “we” and “you”
- Be clear and direct
- Avoid jargon unless necessary

Example:

GOOD:
We help you move with clarity and confidence.

BAD:
Our solution enables transformational capability enhancement.

---

# IMPLEMENTATION RULES

When generating code:

- Keep CSS minimal and intentional
- Prefer:
  - spacing
  - typography
  - layout

Over:
  - effects
  - animation
  - decoration

---

# MOTION (OPTIONAL)

If used:
- Subtle only
- Fade, slide, or reveal
- No bounce, no playful easing

---

# EXAMPLES

### Example 1
User: "build a proposal page"

→ Structured layout
→ Clear sections
→ Orange highlights
→ Strong typography hierarchy

Result: Clean, client-ready proposal interface

---

### Example 2
User: "dashboard for CEO insights"

→ Data-first layout
→ Minimal styling
→ Clear charts
→ Strong headings

Result: Executive-ready dashboard

---

### Example 3
User: "AI transformation page"

→ Forward layout flow
→ Confident messaging
→ Subtle motion
→ Clean visual hierarchy

Result: Modern, premium PwC-style interface