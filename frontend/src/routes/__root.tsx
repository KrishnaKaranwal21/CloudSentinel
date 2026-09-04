import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Outlet, createRootRouteWithContext, HeadContent, Scripts } from "@tanstack/react-router";
import { type ReactNode } from "react";
import appCss from "../styles.css?url";

function ErrorComponent({ error, reset }: { error: Error; reset: () => void }) {
  const router = requireRouter();
  console.error(error);
  return <div className="center-state"><div><h1>HazardWizard could not load this view.</h1><p>Refresh the page or return to the monitored location.</p><button onClick={()=>{router.invalidate();reset();}}>Try again</button></div></div>;
}
function requireRouter(){ return { invalidate: ()=>window.location.reload() }; }

export const Route = createRootRouteWithContext<{ queryClient: QueryClient }>()({
  head: () => ({
    meta: [
      { charSet: "utf-8" },
      { name: "viewport", content: "width=device-width, initial-scale=1" },
      { title: "HazardWizard — CloudSentinel Weather Intelligence" },
      { name: "description", content: "Predictive weather hazard intelligence with AWS anomaly detection, sensor validation and impact awareness." },
      { property: "og:title", content: "HazardWizard — CloudSentinel Weather Intelligence" },
      { property: "og:description", content: "Predict. Detect. Correct. Explain. Assess impact." },
    ],
    links: [
      { rel: "preconnect", href: "https://fonts.googleapis.com" },
      { rel: "preconnect", href: "https://fonts.gstatic.com", crossOrigin: "anonymous" },
      { rel: "stylesheet", href: "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" },
      { rel: "stylesheet", href: appCss },
      { rel: "icon", href: "/favicon.ico", type: "image/x-icon" },
    ],
  }),
  shellComponent: RootShell,
  component: RootComponent,
  errorComponent: ErrorComponent,
});

function RootShell({ children }: { children: ReactNode }) {
  return <html lang="en"><head><HeadContent /></head><body>{children}<Scripts /></body></html>;
}
function RootComponent() {
  const { queryClient } = Route.useRouteContext();
  return <QueryClientProvider client={queryClient}><Outlet /></QueryClientProvider>;
}
