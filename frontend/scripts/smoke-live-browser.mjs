import { chromium } from "@playwright/test";

const appUrl = process.env.CRIMENET_FRONTEND_URL || "http://localhost:3000";
const apiUrl = (process.env.CRIMESENSE_API_URL || "https://api.crimesense.ai").replace(/\/$/, "");
const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const runtimeErrors = [];
const requestPaths = [];
const failedResponses = [];
let holdNextViewport = false;
let releaseHeldViewport;

page.on("console", (message) => {
  if (message.type() === "error" && !message.text().startsWith("Failed to load resource:")) {
    runtimeErrors.push(`console: ${message.text()}`);
  }
});
page.on("pageerror", (error) => runtimeErrors.push(`page: ${error.message}`));
page.on("response", (response) => {
  if (response.status() >= 400) failedResponses.push(`${response.status()} ${response.url()}`);
});
page.on("request", (request) => {
  const url = new URL(request.url());
  if (url.origin === apiUrl) requestPaths.push(`${url.pathname}${url.search}`);
});
await page.route(`${apiUrl}/api/v1/intensity/viewport?*`, async (route) => {
  if (holdNextViewport) {
    holdNextViewport = false;
    await new Promise((resolve) => {
      releaseHeldViewport = resolve;
    });
  }
  await route.continue();
});

const viewportResponse = () =>
  page.waitForResponse((response) => response.url().includes("/api/v1/intensity/viewport?"), {
    timeout: 30_000,
  });

try {
  await page.goto(`${appUrl}/explorer`, { waitUntil: "domcontentloaded" });
  const canvas = page.locator(".map-canvas");
  await page.locator(".map-canvas[data-prediction-layer=ready]").waitFor({ timeout: 30_000 });
  await page.getByText("LIVE API", { exact: true }).waitFor();
  await page.locator(".system-state").filter({ hasText: "SNAPSHOT" }).waitFor();
  const box = await canvas.boundingBox();
  if (!box) throw new Error("Map canvas did not expose a browser bounding box");

  const centerBeforeSearch = await canvas.getAttribute("data-map-center-longitude");
  await page.getByRole("button", { name: "Open command palette" }).click();
  const geocodingResponse = page.waitForResponse(
    (response) => response.url().includes("api.maptiler.com/geocoding/"),
    { timeout: 20_000 },
  );
  await page
    .getByRole("combobox", { name: "Search addresses, places, and CrimeSense commands" })
    .fill("1234 Westheimer Rd, Houston, TX");
  const providerResponse = await geocodingResponse;
  if (!providerResponse.ok()) {
    throw new Error(`MapTiler geocoding failed with ${providerResponse.status()}`);
  }
  const firstSuggestion = page.getByRole("option").first();
  await firstSuggestion.waitFor({ timeout: 10_000 });
  const selectedSuggestion = (await firstSuggestion.textContent())?.trim();
  const searchedViewportResponse = viewportResponse();
  await firstSuggestion.click();
  await searchedViewportResponse;
  await page.locator(".search-destination-marker").waitFor();
  const centerAfterSearch = await canvas.getAttribute("data-map-center-longitude");
  if (!centerAfterSearch || centerAfterSearch === centerBeforeSearch) {
    throw new Error("Selecting a real MapTiler suggestion did not move MapLibre");
  }

  const productionTimeline = await fetch(`${apiUrl}/api/v1/intensity/timeline`).then((response) => {
    if (!response.ok) throw new Error(`Production timeline returned ${response.status}`);
    return response.json();
  });
  const forecastSnapshot = productionTimeline.snapshots.find(
    (snapshot) => snapshot.horizon_hours === 6,
  );
  if (!forecastSnapshot) throw new Error("Production timeline did not expose a +6h forecast");
  const nextForecastSnapshot = productionTimeline.snapshots.find(
    (snapshot) => snapshot.horizon_hours === 7,
  );
  if (!nextForecastSnapshot) throw new Error("Production timeline did not expose a +7h forecast");
  const forecastSlider = page.getByRole("slider", { name: "Forecast hour" });
  await forecastSlider.waitFor({ timeout: 20_000 });
  await forecastSlider.focus();
  for (let step = 0; step < 6; step += 1) await forecastSlider.press("ArrowRight");
  await page.getByText("+6H FORECAST", { exact: true }).waitFor();
  await page
    .locator(
      `.map-canvas[data-snapshot-timestamp="${new Date(forecastSnapshot.valid_utc_hour).toISOString()}"]`,
    )
    .waitFor({ timeout: 30_000 });

  let nativeResolution = Number(await canvas.getAttribute("data-h3-resolution"));
  for (let attempt = 0; nativeResolution !== 9 && attempt < 3; attempt += 1) {
    const response = viewportResponse();
    await page.mouse.move(box.x + box.width * 0.62, box.y + box.height * 0.48);
    await page.mouse.wheel(0, -5000);
    await response;
    await page.waitForTimeout(250);
    nativeResolution = Number(await canvas.getAttribute("data-h3-resolution"));
  }
  if (nativeResolution !== 9) {
    throw new Error(`Expected a drillable native r9 view, received r${nativeResolution}`);
  }

  holdNextViewport = true;
  const nextViewportRequest = page.waitForRequest(
    (request) => request.url().includes("/api/v1/intensity/viewport?"),
    { timeout: 20_000 },
  );
  const nextViewportResponse = viewportResponse();
  await page.mouse.move(box.x + box.width * 0.62, box.y + box.height * 0.48);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * 0.69, box.y + box.height * 0.48, { steps: 10 });
  await page.mouse.up();
  await nextViewportRequest;
  if ((await canvas.getAttribute("data-prediction-layer")) !== "ready") {
    throw new Error("The previous H3 layer disappeared while the panned viewport was loading");
  }
  releaseHeldViewport?.();
  await nextViewportResponse;
  await page.locator(".map-canvas[data-prediction-layer=ready]").waitFor({ timeout: 20_000 });

  const reverseGeocodingResponse = page.waitForResponse(
    (response) =>
      response.url().includes("api.maptiler.com/geocoding/") &&
      !response.url().includes(encodeURIComponent("1234 Westheimer")),
    { timeout: 20_000 },
  );
  await page.mouse.click(box.x + box.width * 0.62, box.y + box.height * 0.48);
  await page.getByText("H3 CELL INSPECTOR").waitFor({ timeout: 20_000 });
  const locationProviderResponse = await reverseGeocodingResponse;
  if (!locationProviderResponse.ok()) {
    throw new Error(`MapTiler reverse geocoding failed with ${locationProviderResponse.status()}`);
  }
  await page.waitForFunction(
    () => document.querySelector(".inspector-header strong")?.textContent?.includes("Houston"),
    undefined,
    { timeout: 20_000 },
  );
  const inspectorLocation = (await page.locator(".inspector-header strong").textContent())?.trim();
  if (!inspectorLocation || inspectorLocation.includes("Chicago")) {
    throw new Error(
      `Clicked-cell location incorrectly followed sidebar city: ${inspectorLocation}`,
    );
  }
  await page.getByText("FORECAST INFERENCE", { exact: true }).waitFor({ timeout: 120_000 });
  await page.getByText("EVENT INTENSITY", { exact: true }).waitFor();
  await page.getByRole("button", { name: "INTENSITY", exact: true }).click();
  await page.getByRole("button", { name: "VIEW ALL 87 →" }).click();
  await page.getByRole("dialog").waitFor();
  const markRows = await page.locator(".types-table > button").count();
  if (markRows !== 87) throw new Error(`Expected 87 inspector rows, received ${markRows}`);
  const forecastMarkPath = [...requestPaths]
    .reverse()
    .find((path) => path.startsWith("/api/v1/predict/cell/"));
  const markValidUtcHour = forecastMarkPath
    ? new URL(forecastMarkPath, apiUrl).searchParams.get("valid_utc_hour")
    : null;
  if (markValidUtcHour !== forecastSnapshot.valid_utc_hour) {
    throw new Error("Selected-cell inference did not use the map's +6h UTC forecast hour");
  }
  await page.getByRole("button", { name: "Close all crime types" }).click();
  const inspectedCellBeforeForecastStep = await page
    .locator(".inspector-header strong")
    .textContent();
  const updatedCellResponse = page.waitForResponse(
    (response) => {
      if (!response.url().includes("/api/v1/predict/cell/")) return false;
      return (
        new URL(response.url()).searchParams.get("valid_utc_hour") ===
        nextForecastSnapshot.valid_utc_hour
      );
    },
    { timeout: 120_000 },
  );
  await forecastSlider.focus();
  await forecastSlider.press("ArrowRight");
  await page.getByText("+7H FORECAST", { exact: true }).waitFor();
  await page
    .locator(
      `.map-canvas[data-snapshot-timestamp="${new Date(nextForecastSnapshot.valid_utc_hour).toISOString()}"]`,
    )
    .waitFor({ timeout: 30_000 });
  const refreshedCellResponse = await updatedCellResponse;
  if (!refreshedCellResponse.ok()) {
    throw new Error(`Updated cell inference returned ${refreshedCellResponse.status()}`);
  }
  await page.getByText("H3 CELL INSPECTOR").waitFor();
  await page.getByText("EVENT INTENSITY", { exact: true }).waitFor({ timeout: 120_000 });
  const inspectedCellAfterForecastStep = await page
    .locator(".inspector-header strong")
    .textContent();
  if (
    !inspectedCellBeforeForecastStep ||
    inspectedCellAfterForecastStep !== inspectedCellBeforeForecastStep
  ) {
    throw new Error("Forecast stepping did not retain the selected H3 cell inspector");
  }
  await page.keyboard.press("Escape");

  let coarseResolution = 9;
  for (let attempt = 0; coarseResolution > 5 && attempt < 8; attempt += 1) {
    const response = viewportResponse();
    await page.mouse.move(box.x + box.width * 0.62, box.y + box.height * 0.48);
    await page.mouse.wheel(0, 5000);
    await response;
    await page.waitForTimeout(250);
    coarseResolution = Number(await canvas.getAttribute("data-h3-resolution"));
  }
  if (coarseResolution > 5) {
    throw new Error(`Expected a national/regional coarse view, received r${coarseResolution}`);
  }
  if ((await canvas.getAttribute("data-prediction-layer")) !== "ready") {
    throw new Error("The coarse H3 LOD surface did not render");
  }
  if (await page.getByText("Zoom in to load live H3 predictions", { exact: true }).isVisible()) {
    throw new Error("The national LOD view incorrectly displayed the legacy zoom-in state");
  }

  const coarsePoints = [
    [0.5, 0.48],
    [0.62, 0.48],
    [0.4, 0.42],
    [0.7, 0.55],
  ];
  let coarseHoverPoint;
  for (const [xRatio, yRatio] of coarsePoints) {
    await page.mouse.move(box.x + box.width * xRatio, box.y + box.height * yRatio);
    await page.waitForTimeout(150);
    const tooltipText = (await page.locator(".map-tooltip").textContent()) ?? "";
    if (
      tooltipText.includes("mean r9-cell intensity") &&
      tooltipText.includes("modeled r9 cells")
    ) {
      coarseHoverPoint = [xRatio, yRatio];
      break;
    }
  }
  if (!coarseHoverPoint) throw new Error("Coarse LOD tooltip metadata did not render");

  const markRequestsBeforeCoarseClick = requestPaths.filter((path) =>
    path.startsWith("/api/v1/predict/cell/"),
  ).length;
  let coarseDrillDownObserved = false;
  for (const [xRatio, yRatio] of [coarseHoverPoint, ...coarsePoints]) {
    const response = page.waitForResponse(
      (candidate) => candidate.url().includes("/api/v1/intensity/viewport?"),
      { timeout: 5_000 },
    );
    await page.mouse.click(box.x + box.width * xRatio, box.y + box.height * yRatio);
    try {
      await response;
      coarseDrillDownObserved = true;
      break;
    } catch {
      // Some national-view coordinates fall outside modeled land cells.
    }
  }
  if (!coarseDrillDownObserved) throw new Error("No coarse H3 cell accepted a drill-down click");
  await page.waitForTimeout(500);
  const drilledResolution = Number(await canvas.getAttribute("data-h3-resolution"));
  const markRequestsAfterCoarseClick = requestPaths.filter((path) =>
    path.startsWith("/api/v1/predict/cell/"),
  ).length;
  if (markRequestsAfterCoarseClick !== markRequestsBeforeCoarseClick) {
    throw new Error("A coarse-cell click incorrectly invoked mark inference");
  }
  if (await page.getByText("H3 CELL INSPECTOR").isVisible()) {
    throw new Error("A coarse-cell click left a stale r9 inspector visible");
  }

  if (runtimeErrors.length) throw new Error(runtimeErrors.join("\n"));
  if (failedResponses.length) throw new Error(failedResponses.join("\n"));
  console.log(
    JSON.stringify(
      {
        live_geocoding: true,
        selected_geocoding_suggestion: selectedSuggestion,
        search_triggered_viewport_inference: true,
        selected_forecast_horizon: forecastSnapshot.horizon_hours,
        forecast_surface_timestamp: forecastSnapshot.valid_utc_hour,
        forecast_cell_timestamp_consistent: true,
        forecast_step_retained_inspector: true,
        native_resolution: nativeResolution,
        native_selected_cell_inference: true,
        clicked_cell_location: inspectorLocation,
        clicked_cell_location_independent_of_sidebar: true,
        mark_rows: markRows,
        viewport_refetched_after_pan: true,
        national_resolution: coarseResolution,
        national_surface_rendered: true,
        coarse_tooltip_metadata: true,
        coarse_click_mark_requests: 0,
        drill_down_resolution: drilledResolution,
        api_paths: [...new Set(requestPaths.map((path) => path.split("?")[0]))],
      },
      null,
      2,
    ),
  );
} finally {
  await browser.close();
}
