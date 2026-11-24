import { useEffect, useMemo, useState } from 'react'
import { cite, pct, STATUS, TOOLS, trace } from '../lib.js'

function Reply({ text, sections }) {
  const [open, setOpen] = useState(null)
  return (
    <div className="reply">
      <h3>Reply</h3>
      <p>
        {cite(text).map((p, i) => p.id
          ? <button key={i} type="button" className="cite" aria-expanded={open === p.id} onClick={() => setOpen(open === p.id ? null : p.id)}>{sections[p.id]?.title ?? p.id}</button>
          : <span key={i}>{p.text}</span>)}
      </p>
      {open && <blockquote className="source">{sections[open]?.text}</blockquote>}
    </div>
  )
}

function Step({ s }) {
  if (s.kind === 'triage') return <><b>Read as “{s.intent}”</b><span>{pct(s.confidence)} sure, {s.model}</span></>
  if (s.kind === 'safety') return <><b>Stopped for safety</b><span>The message says “{s.matched}”. No model was asked.</span></>
  if (s.kind === 'approval') return <><b>Waiting for a person</b><span>£{Number(s.args.amount).toFixed(2)} on {s.args.ref}. {s.args.reason}.</span></>
  if (s.kind === 'decision') return <><b>{s.approved ? 'You approved it' : 'You turned it down'}</b></>
  if (s.kind === 'handover') return <><b>Handed to a person</b><span>{s.reason}</span></>
  if (s.kind === 'reply') return <><b>Wrote a reply</b><span>{s.cited.length ? 'Every citation was retrieved' : 'No policy claims'}</span></>
  if (s.kind === 'result') return <><b>{s.error ? 'Refused' : 'Done'}</b><span>{s.data.error ?? JSON.stringify(s.data)}</span></>
  const r = s.result
  const detail = !r ? '' : r.error ? r.data.error
    : s.tool === 'search_help' ? (r.data.found ? r.data.sections.map((x) => x.title).join('; ') : 'Nothing close enough')
      : s.tool === 'list_orders' ? r.data.orders.map((o) => o.ref).join(', ')
        : r.data.item ? `${r.data.ref}, ${r.data.item}, ${r.data.status}`
          : r.data.label ? `${r.data.label}, ${r.data.days_left} days left` : r.data.status ? `${r.data.ref} is ${r.data.status}` : ''
  return <><b>{TOOLS[s.tool] ?? s.tool}{r?.error ? ', refused' : ''}</b><span>{detail}</span></>
}

const tone = (s) => ['safety', 'handover'].includes(s.kind) || s.result?.error || s.error ? 'bad' : ['approval'].includes(s.kind) ? 'wait' : 'ok'

export default function Inbox({ data }) {
  const [at, setAt] = useState(0)
  const [decided, setDecided] = useState({})
  useEffect(() => {
    document.title = 'Open Desk'
  }, [])
  const tickets = useMemo(() => data.tickets.map((t, i) => {
    const d = decided[i] && t.after[decided[i]]
    return d ? { ...t, status: d.status, steps: [...t.steps, ...d.steps], decided: decided[i] } : t
  }), [data, decided])
  const t = tickets[at]
  const reply = t.steps.findLast((s) => s.kind === 'reply')
  const ask = t.status === 'waiting' ? t.steps.findLast((s) => s.kind === 'approval') : null

  return (
    <div className="desk">
      <nav className="list" aria-label="Messages">
        <h1>Inbox</h1>
        <ul>
          {tickets.map((x, i) => (
            <li key={x.body}>
              <button type="button" aria-current={i === at ? 'true' : undefined} onClick={() => setAt(i)}>
                <span className="list__who">{x.name}<i className={`pill pill--${x.status}`}>{STATUS[x.status]}</i></span>
                <span className="list__body">{x.body}</span>
              </button>
            </li>
          ))}
        </ul>
      </nav>
      <article className="thread" aria-label="Message">
        <header><h2>{t.name}</h2><p>{t.email}</p></header>
        <p className="thread__body">{t.body}</p>
        {ask && (
          <section className="ask" aria-label="Decision">
            <h3>Refund £{Number(ask.args.amount).toFixed(2)} on {ask.args.ref}?</h3>
            <p>{ask.args.reason}. The agent cannot do this itself.</p>
            <div>
              <button type="button" className="btn" onClick={() => setDecided({ ...decided, [at]: 'approve' })}>Approve refund</button>
              <button type="button" className="btn btn--quiet" onClick={() => setDecided({ ...decided, [at]: 'reject' })}>Turn down</button>
            </div>
          </section>
        )}
        <div aria-live="polite">
          {reply && <Reply key={`${at}${t.decided}`} text={reply.text} sections={data.sections} />}
          {t.status === 'with_person' && <p className="held">A person is dealing with this. {t.steps.findLast((s) => s.kind === 'handover')?.reason}.</p>}
        </div>
      </article>
      <aside className="trace" aria-label="What the agent did">
        <h2>What happened</h2>
        <ol>
          {trace(t.steps).map((s, i) => <li key={i} className={`trace__${tone(s)}`}><Step s={s} /></li>)}
        </ol>
      </aside>
    </div>
  )
}
