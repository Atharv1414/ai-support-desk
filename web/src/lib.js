export const pct = (v, d = 0) => `${(100 * v).toFixed(d)}%`
export const STATUS = { answered: 'Answered', waiting: 'Needs your decision', with_person: 'With a person', new: 'New' }
export const TOOLS = {
  list_orders: 'Listed the customer’s orders', find_order: 'Looked up the order', search_help: 'Searched the help centre',
  cancel_order: 'Cancelled the order', start_return: 'Started a return', issue_refund: 'Asked to refund', escalate: 'Handed over',
}

/** Pair each tool call with its result, so the trace reads as one line per action. */
export function trace(steps) {
  const out = []
  for (const s of steps) {
    if (s.kind === 'result') {
      const open = out.findLast((x) => x.kind === 'call' && x.tool === s.tool && !x.result)
      if (open) {
        open.result = s
        continue
      }
    }
    out.push({ ...s })
  }
  return out
}

/** Split a reply into text and citations. */
export function cite(text) {
  return text.split(/(\[[a-z0-9-]+#[a-z0-9-]+\])/).filter(Boolean).map((part) =>
    /^\[.+#.+\]$/.test(part) ? { id: part.slice(1, -1) } : { text: part })
}
