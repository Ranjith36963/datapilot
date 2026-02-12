# DataPilot Frontend

Next.js 16 web application with TypeScript, Tailwind CSS, and shadcn/ui.

## Pages

1. `/` — Upload (drag-drop, data preview, profile card, domain detection badge with explainability, "Start Exploring" CTA)
2. `/explore` — Chat interface (AI Insights auto-pilot section at top, sidebar with data info, streaming chat, embedded charts)
3. `/visualize` — Chart builder (type selector, column dropdowns, live preview)
4. `/export` — Report generator (analysis selector, format picker, preview, download)

## Design System

- **Font**: Inter
- **Palette**: slate/blue (dark/light mode via next-themes)
- **Components**: shadcn/ui (Radix primitives + Tailwind)
- **Charts**: Plotly.js via react-plotly.js
- **Tables**: @tanstack/react-table
- **Icons**: lucide-react
- **File upload**: react-dropzone

## Conventions

- App Router (app/ directory)
- Server components by default, "use client" only when needed
- API calls to backend at `NEXT_PUBLIC_API_URL` (default: http://localhost:8000)
- Tailwind for all styling (no CSS modules)
- Components in `src/components/`, hooks in `src/hooks/`, utils in `src/lib/`

## Running

```bash
npm run dev    # development
npm run build  # production build
npm run lint   # eslint
```
