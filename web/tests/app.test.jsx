import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it } from 'vitest'
import App from '../src/App.jsx'
import { cite, trace } from '../src/lib.js'
import data from '../src/recorded.json'

const open = (path = '/') => render(<MemoryRouter initialEntries={[path]}><App data={data} /></MemoryRouter>)
const list = () => screen.getByRole('navigation', { name: 'Messages' })
const pick = (u, text) => u.click(within(list()).getByRole('button', { name: new RegExp(text) }))

describe('helpers', () => {
  it('pairs calls with results, even repeated ones', () => {
    const t = trace([{ kind: 'call', tool: 'a' }, { kind: 'call', tool: 'a' }, { kind: 'result', tool: 'a', n: 1 }, { kind: 'result', tool: 'a', n: 2 }])
    expect(t).toHaveLength(2)
    expect(t.map((x) => x.result.n).sort()).toEqual([1, 2])
  })
  it('splits citations out of a reply', () => {
    expect(cite('Yes. [a-b#c-d] Bye')).toEqual([{ text: 'Yes. ' }, { id: 'a-b#c-d' }, { text: ' Bye' }])
    expect(cite('No [links] here')).toEqual([{ text: 'No [links] here' }])
  })
})

describe('what was recorded', () => {
  it('cites only sections that exist', () => {
    for (const t of data.tickets)
      for (const s of [...t.steps, ...(t.after ? Object.values(t.after).flatMap((a) => a.steps) : [])])
        for (const id of s.cited ?? []) expect(data.sections).toHaveProperty([id])
  })
  it('never pays without a decision, and pays only in the approved branch', () => {
    const paid = (steps) => steps.some((s) => s.kind === 'result' && s.tool === 'issue_refund' && s.data.refunded)
    const waiting = data.tickets.filter((t) => t.after)
    expect(waiting.length).toBeGreaterThan(0)
    for (const t of data.tickets) expect(paid(t.steps)).toBe(false)
    for (const t of waiting) {
      expect(paid(t.after.approve.steps)).toBe(true)
      expect(paid(t.after.reject.steps)).toBe(false)
    }
  })
  it('never shows one customer another’s order', () => {
    const own = { 'ada@example.com': ['HF-20418', 'HF-20533'], 'ben@example.com': ['HF-21007'], 'cal@example.com': ['HF-21090'], 'dee@example.com': ['HF-21555'], 'eli@example.com': ['HF-22101'] }
    for (const t of data.tickets)
      for (const s of t.steps.filter((x) => x.kind === 'result' && !x.error))
        for (const ref of JSON.stringify(s.data).match(/HF-\d{5}/g) ?? []) expect(own[t.email]).toContain(ref)
  })
})

describe('inbox', () => {
  it('lets a person approve a refund and shows what follows', async () => {
    const u = userEvent.setup()
    open()
    await pick(u, 'where\'s my money')
    expect(screen.getByRole('region', { name: 'Decision' })).toBeInTheDocument()
    await u.click(screen.getByRole('button', { name: 'Approve refund' }))
    expect(screen.queryByRole('region', { name: 'Decision' })).not.toBeInTheDocument()
    expect(screen.getByText('You approved it')).toBeInTheDocument()
    expect(screen.getByText(/has been approved/)).toBeInTheDocument()
  })
  it('turning it down pays nothing and says so', async () => {
    const u = userEvent.setup()
    open()
    await pick(u, 'where\'s my money')
    await u.click(screen.getByRole('button', { name: 'Turn down' }))
    expect(screen.getByText('You turned it down')).toBeInTheDocument()
    expect(screen.queryByText(/has been approved/)).not.toBeInTheDocument()
  })
  it('shows that danger stops before any model', async () => {
    const u = userEvent.setup()
    open()
    await pick(u, 'smoke')
    expect(screen.getByText('Stopped for safety')).toBeInTheDocument()
    expect(screen.getByText(/No model was asked/)).toBeInTheDocument()
  })
  it('opens the source behind a citation', async () => {
    const u = userEvent.setup()
    open()
    await pick(u, 'descale')
    const c = screen.getByRole('button', { name: /kettle: Descaling/ })
    await u.click(c)
    expect(c).toHaveAttribute('aria-expanded', 'true')
    expect(screen.getByText(/white vinegar/, { selector: 'blockquote' })).toBeInTheDocument()
  })
})

it('shows every model, its gate, and the one in use', () => {
  open('/models')
  const table = screen.getByRole('region', { name: 'Model comparison' })
  expect(within(table).getAllByRole('row')).toHaveLength(data.models.length + 1)
  expect(within(table).getByText('in use')).toBeInTheDocument()
  expect(screen.getAllByRole('img', { name: /calibration error/ })).toHaveLength(data.models.length)
})

it('reports search on the half it was not tuned on', () => {
  open('/search')
  expect(screen.getByText(`${Math.round(100 * data.retrieval.test.refuses_unanswerable)}%`)).toBeInTheDocument()
})
