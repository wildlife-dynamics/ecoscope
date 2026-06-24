# Ecoscope Nav Handoff

Copy the top nav from the ecoscope repo into another project.

**Source repo:** `~/Desktop/ecoscope`
**Branch:** `700-ecoscopeio-tweaks`
**Files:** `html/index.html`, `html/styles.css`, `html/site.js`

---

## What to extract

### HTML
From `html/index.html`, copy:
1. The `<header>` block (lines ~19–58) — contains the desktop nav and mobile overlay
2. The Google Fonts `<link>` tag in `<head>` (not present yet — see CSS note below)

### CSS
From `html/styles.css`, copy all rules related to:
- `header`, `header nav`
- `.logo-link-raised`
- `.nav-links`, `.nav-links a`
- `.nav-dropdown`, `.nav-dropdown-toggle`, `.nav-dropdown-toggle::after`
- `.nav-dropdown.open`
- `.nav-submenu`, `.nav-submenu[hidden]`, `.nav-submenu .nav-submenu-link`
- `.hamburger-btn`, `.hamburger-btn svg`, `.close-icon-initial`
- `.mobile-overlay`, `.mobile-overlay.open`, `.mobile-overlay a`
- `.mobile-menu-links`, `.mobile-menu-group`, `.mobile-menu-label`
- `.mobile-overlay .mobile-submenu-link`
- The `@media (min-width: 768px)` block that shows `.nav-links` and hides `.hamburger-btn`
- The `@import` for Manrope at the top of the file

### JS
From `html/site.js`, copy everything **except** `addPagePrefetching()` and `prefetchSameSitePage()` — those are ecoscope-specific. The nav needs:
- `toggleMobileMenu()` and `closeMobileMenu()`
- `setDropdownOpen()` and `closeAllDropdowns()`
- All the event listeners wired to `.nav-dropdown`, `.hamburger-btn`, `.mobile-overlay a`

---

## Nav structure

- **Logo** — far left
- **Nav links** — immediately right of logo (desktop only)
  - Web App → `https://app.ecoscope.io/login`
  - Desktop App → `https://app.ecoscope.io/download`
  - Develop (dropdown) → SDK, Core (both open in new tab)
  - Help (dropdown) → Help Center, Ecoscope Community, Web Release Notes, Desktop Release Notes
- **Hamburger / X** — far right (mobile only)

## Mobile menu
- Full-screen dark overlay with blur
- All items left-aligned
- Groups (Develop, Help) shown as labeled sections with indented sub-links
- Hamburger icon toggles to X in the same position (far right)

## Font
Manrope via Google Fonts — add to `<head>`:
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap">
```
CSS variable: `--font-sans: 'Manrope', system-ui, sans-serif;`

## Dropdown style
Black 50% opacity with backdrop blur and white text. Defined in `.nav-submenu`.
