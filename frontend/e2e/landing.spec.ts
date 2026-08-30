import { expect, test } from "@playwright/test";

test("landing page explains the platform without overflow", async ({ page }, testInfo) => {
  const runtimeErrors: string[] = [];
  page.on("pageerror", (error) => runtimeErrors.push(error.message));
  await page.emulateMedia({ reducedMotion: "reduce" });

  await page.goto("/");
  await expect(page.getByText("CRIMESENSE").first()).toBeVisible();
  await expect(page.getByText("POWERED BY CRIMENET")).toHaveCount(1);
  await expect(page.getByRole("heading", { name: /spatiotemporal intelligence/i })).toBeVisible();
  await expect(
    page.getByText(/More than 17 million municipal and county crime records/),
  ).toBeVisible();
  await expect(page.getByText("Spark / Databricks")).toBeVisible();
  await expect(page.getByRole("link", { name: /open explorer/i }).first()).toHaveAttribute(
    "href",
    "/explorer",
  );

  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - window.innerWidth,
  );
  expect(overflow).toBeLessThanOrEqual(1);
  expect(runtimeErrors).toEqual([]);
  await page.screenshot({ path: testInfo.outputPath("landing.png"), fullPage: true });
});
