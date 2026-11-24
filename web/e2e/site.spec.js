import AxeBuilder from '@axe-core/playwright'
import { expect, test } from '@playwright/test'

for (const scheme of ['light', 'dark']) {
  test(`every page is accessible and never scrolls sideways: ${scheme}`, async ({ page }) => {
    const errors = []
    page.on('pageerror', (e) => errors.push(e.message))
    await page.emulateMedia({ colorScheme: scheme, reducedMotion: 'reduce' })
    for (const path of ['/', '/models', '/search']) {
      await page.goto(path)
      await expect(page.locator('h1')).toBeVisible()
      expect(await page.evaluate(() => document.documentElement.scrollWidth - innerWidth), path).toBeLessThanOrEqual(0)
      const axe = await new AxeBuilder({ page }).analyze()
      expect(axe.violations.map((v) => `${path} ${v.id}: ${v.nodes[0].html.slice(0, 160)} ${v.nodes[0].failureSummary}`)).toEqual([])
    }
    expect(errors).toEqual([])
  })
}

test('every message opens, and a refund can be approved by keyboard', async ({ page }) => {
  await page.goto('/')
  const items = page.getByRole('navigation', { name: 'Messages' }).getByRole('button')
  const n = await items.count()
  for (let i = 0; i < n; i++) {
    await items.nth(i).click()
    await expect(page.locator('.trace li').first()).toBeVisible()
  }
  await items.first().press('Enter')
  await page.getByRole('button', { name: 'Approve refund' }).press('Enter')
  await expect(page.getByText('You approved it')).toBeVisible()
})
