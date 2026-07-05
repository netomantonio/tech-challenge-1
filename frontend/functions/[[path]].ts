type ApiFetcher = (request: Request) => Promise<Response>;

export async function proxyRequest(request: Request, fetchApi: ApiFetcher) {
  try {
    const upstream = await fetchApi(request);
    const headers = new Headers(upstream.headers);
    headers.set("X-Content-Type-Options", "nosniff");
    headers.set("Referrer-Policy", "no-referrer");
    headers.set("Cache-Control", "no-store");
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers,
    });
  } catch (error) {
    console.error(
      JSON.stringify({
        event: "api_proxy_failed",
        errorType: error instanceof Error ? error.name : "UnknownError",
      }),
    );
    return Response.json(
      { detail: "A API está temporariamente indisponível." },
      { status: 503 },
    );
  }
}

export const onRequest: PagesFunction<Env> = async (context) => {
  return proxyRequest(context.request, (request) => context.env.API.fetch(request));
};
