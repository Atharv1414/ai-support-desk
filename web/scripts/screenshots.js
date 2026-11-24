// Usage: npm run build, vite preview on port 4173, then node scripts/screenshots.js <folder>
import process from 'node:process'
import { chromium } from '@playwright/test'

const b = await chromium.launch({ channel: process.env.PW_CHANNEL })
for (const [name, path, w, h, scheme] of [['inbox', '/', 1360, 800, 'light'], ['models', '/models', 1280, 900, 'light'], ['phone', '/', 390, 844, 'dark']]) {
  const c = await b.newContext({ viewport: { width: w, height: h }, colorScheme: scheme, reducedMotion: 'reduce' })
  const p = await c.newPage()
  await p.goto(`http://localhost:4173${path}`)
  await p.waitForTimeout(400)
  await p.screenshot({ path: `${process.argv[2]}/${name}.png`, fullPage: name !== 'inbox' })
  await c.close()
}
await b.close()
