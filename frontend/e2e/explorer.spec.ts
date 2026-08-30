import { expect, test } from "@playwright/test";

test("primary explorer and model flow", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  const runtimeErrors: string[] = [];
  page.on("pageerror", (error) => runtimeErrors.push(error.message));
  page.on("console", (message) => {
    const source = message.location().url;
    if (message.type() === "error" && !source.endsWith("/favicon.ico")) {
      runtimeErrors.push(`${source || "browser"}: ${message.text()}`);
    }
  });

  await page.goto("/explorer");
  await expect(page.getByText("DEVELOPMENT FIXTURE")).toBeVisible({ timeout: 15_000 });
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-map-loaded", "true", {
    timeout: 15_000,
  });
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-prediction-layer", "ready", {
    timeout: 15_000,
  });
  await expect(page.locator("canvas").first()).toBeVisible({ timeout: 15_000 });
  await expect(
    page.locator(".map-legend").getByText("PREDICTED INTENSITY", { exact: true }),
  ).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("explorer.png"), fullPage: true });

  const timeBeforeBasemapSwitch = new URL(page.url()).searchParams.get("time");
  await page.getByRole("button", { name: "SATELLITE" }).click();
  await expect(page.getByRole("button", { name: "SATELLITE" })).toHaveAttribute(
    "aria-pressed",
    "true",
  );
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-basemap-mode", "satellite");
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-satellite-layer-visible", "true");
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-satellite-loaded", "true", {
    timeout: 15_000,
  });
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-prediction-layer", "ready");
  expect(new URL(page.url()).searchParams.get("time")).toBe(timeBeforeBasemapSwitch);
  await expect(page.locator(".maplibregl-ctrl-attrib")).toBeVisible();
  await expect(page.locator(".maplibregl-ctrl-attrib")).toContainText("MapTiler");
  await page.screenshot({ path: testInfo.outputPath("explorer-satellite.png"), fullPage: true });

  const viewport = page.viewportSize();
  if (viewport && testInfo.project.name === "desktop-1440") {
    await page.mouse.click(viewport.width * 0.55, viewport.height * 0.57);
    await expect(page.getByText("H3 CELL INSPECTOR")).toBeVisible();
    await expect(page.getByText("EVENT INTENSITY")).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath("satellite-selected.png"), fullPage: true });
  }
  await page.getByRole("button", { name: "3D" }).click();
  await expect(page.getByRole("button", { name: "3D" })).toHaveClass(/active/);
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-prediction-layer", "ready");
  await page.getByRole("button", { name: "DARK" }).click();
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-basemap-mode", "dark");
  if (testInfo.project.name === "desktop-1440") {
    await expect(page.getByText("H3 CELL INSPECTOR")).toBeVisible();
    await page.keyboard.press("Escape");
  }

  if (viewport) {
    await page.mouse.click(viewport.width * 0.55, viewport.height * 0.57);
    await expect(page.getByText("H3 CELL INSPECTOR")).toBeVisible();
    await expect(page.getByText("CRIME MIX")).toBeVisible();
    await expect(page.getByText("FIXTURE DATA")).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath("inspector.png"), fullPage: true });
    if (testInfo.project.name === "desktop-1440") {
      const timeBefore = new URL(page.url()).searchParams.get("time");
      const comparisonBar = page.locator(".crime-mix .distribution-track i").nth(1);
      const barWidthBefore = await comparisonBar.getAttribute("style");
      await page.keyboard.press("ArrowRight");
      await expect.poll(() => new URL(page.url()).searchParams.get("time")).not.toBe(timeBefore);
      await expect.poll(() => comparisonBar.getAttribute("style")).not.toBe(barWidthBefore);
      await page.getByRole("button", { name: "INTENSITY", exact: true }).click();
      await expect(page.getByText("λfamily · events / cell / hour")).toBeVisible();
      await page.getByRole("button", { name: "VIEW ALL 87 →" }).click();
      await expect(page.getByRole("dialog")).toBeVisible();
      await page.getByPlaceholder("Search modeled crime type...").fill("theft from vehicle");
      await expect(page.getByRole("dialog").getByText("Theft From Vehicle")).toBeVisible();
      await page.screenshot({ path: testInfo.outputPath("all-87-types.png"), fullPage: true });
      await page.getByRole("button", { name: "Close all crime types" }).click();
    }
    await page.keyboard.press("Escape");
  }

  await page.goto("/model");
  await expect(page.getByText("SUPPORTED GEOGRAPHIES")).toBeVisible();
  await expect(page.getByText("Evaluation artifact API required")).toBeVisible();
  expect(runtimeErrors).toEqual([]);
});

test("satellite tile failure returns safely to dark", async ({ page }, testInfo) => {
  test.skip(
    testInfo.project.name !== "desktop-1440",
    "One deterministic recovery check is sufficient",
  );
  await page.route("https://api.maptiler.com/maps/satellite-v4/**", (route) => {
    if (route.request().url().includes("tiles.json")) return route.continue();
    return route.abort();
  });
  await page.goto("/explorer");
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-map-loaded", "true", {
    timeout: 15_000,
  });
  await page.getByRole("button", { name: "SATELLITE" }).click();
  await expect(
    page.getByText("Satellite imagery unavailable. Using standard basemap."),
  ).toBeVisible({
    timeout: 15_000,
  });
  await expect(page.getByRole("button", { name: "DARK" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-prediction-layer", "ready");
});

test("prediction surface survives a basemap style failure", async ({ page }, testInfo) => {
  test.skip(
    testInfo.project.name !== "desktop-1440",
    "One deterministic recovery check is sufficient",
  );
  await page.route("https://api.maptiler.com/**", (route) => route.abort());
  await page.goto("/explorer");
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-map-fallback", "true", {
    timeout: 15_000,
  });
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-prediction-layer", "ready", {
    timeout: 15_000,
  });
  const viewport = page.viewportSize();
  if (viewport) await page.mouse.click(viewport.width * 0.55, viewport.height * 0.57);
  await expect(page.getByText("H3 CELL INSPECTOR")).toBeVisible();
});

test("location search selects a suggestion and moves the existing map", async ({ page }, testInfo) => {
  test.skip(
    testInfo.project.name !== "desktop-1440",
    "One deterministic geocoding browser check is sufficient",
  );
  await page.route("https://api.maptiler.com/geocoding/**", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        type: "FeatureCollection",
        query: ["1234", "westheimer"],
        attribution: "MapTiler",
        features: [
          {
            type: "Feature",
            id: "address.mock-westheimer",
            text: "Westheimer Road",
            address: "1234",
            place_name: "1234 Westheimer Road, Houston, Texas 77006, United States",
            place_type: ["address"],
            place_type_name: ["Address"],
            center: [-95.3936, 29.7411],
            geometry: { type: "Point", coordinates: [-95.3936, 29.7411] },
            properties: { ref: "mock", country_code: "us" },
            relevance: 1,
          },
        ],
      }),
    });
  });
  await page.goto("/explorer");
  await expect(page.locator(".map-canvas")).toHaveAttribute("data-map-loaded", "true", {
    timeout: 15_000,
  });
  const longitudeBefore = await page
    .locator(".map-canvas")
    .getAttribute("data-map-center-longitude");
  await page.getByRole("button", { name: "Open command palette" }).click();
  await page
    .getByRole("combobox", { name: "Search addresses, places, and CrimeNet commands" })
    .fill("1234 Westheimer Rd, Houston TX");
  await expect(page.getByText("1234 Westheimer Road", { exact: true })).toBeVisible();
  await page.getByRole("option", { name: /1234 Westheimer Road/ }).click();
  await expect(page.locator(".command-dialog")).not.toBeVisible();
  await expect(page.locator(".search-destination-marker")).toBeVisible();
  await expect
    .poll(() => page.locator(".map-canvas").getAttribute("data-map-center-longitude"), {
      timeout: 5_000,
    })
    .not.toBe(longitudeBefore);
  await expect(page.locator(".map-canvas")).toHaveAttribute(
    "data-map-center-longitude",
    "-95.39360",
  );
});
