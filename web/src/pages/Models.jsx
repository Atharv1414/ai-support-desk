import { useEffect, useState } from 'react'
import { pct } from '../lib.js'

const CHECK = { true: '✓', false: '✗' }

function Reliability({ models }) {
  const s = 240
  const p = 34
  const x = (v) => p + v * (s - p - 8)
  const y = (v) => s - p - v * (s - p - 8)
  return (
    <div className="small">
      {models.map((m) => (
        <figure key={m.version}>
          <figcaption>{m.version}</figcaption>
          <svg viewBox={`0 0 ${s} ${s}`} role="img" aria-label={`${m.version}: calibration error ${pct(m.metrics.ece, 1)}`}>
            {[0, 0.5, 1].map((v) => (
              <g key={v}>
                <path className="grid" d={`M${x(0)} ${y(v)}H${x(1)}`} />
                <text x={p - 6} y={y(v) + 4} textAnchor="end">{pct(v)}</text>
                <text x={x(v)} y={s - 14} textAnchor={v === 1 ? 'end' : 'middle'}>{pct(v)}</text>
              </g>
            ))}
            <path className="ideal" d={`M${x(0)} ${y(0)}L${x(1)} ${y(1)}`} />
            {m.metrics.reliability.map((b) => <circle key={b.confidence} cx={x(b.confidence)} cy={y(b.accuracy)} r={2.5 + Math.sqrt(b.n) * 0.9} />)}
          </svg>
          <p>Calibration error {pct(m.metrics.ece, 1)}</p>
        </figure>
      ))}
    </div>
  )
}

function Confusion({ m, labels }) {
  const c = m.metrics.confusion
  return (
    <div className="scroll" role="region" aria-label="Confusion table" tabIndex={0}>
      <table className="heat">
        <caption>{m.version}: rows are the truth, columns are the answer</caption>
        <thead><tr><td />{labels.map((l) => <th key={l} scope="col">{l}</th>)}</tr></thead>
        <tbody>
          {c.map((row, i) => (
            <tr key={labels[i]}>
              <th scope="row">{labels[i]}</th>
              {row.map((n, j) => <td key={labels[j]} className={i === j ? 'hit' : n ? 'miss' : ''} style={{ '--n': n / 12 }}>{n || ''}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function Models({ data }) {
  const labels = Object.keys(data.intents)
  const [version, setVersion] = useState(data.champion)
  const m = data.models.find((x) => x.version === version)
  useEffect(() => {
    document.title = 'Models – Open Desk'
  }, [])
  return (
    <div className="page">
      <h1>Which model reads the mail</h1>
      <p className="lead">Three models sort messages into ten topics. They are tested on {m.test_size} messages written by hand, and a new one takes over only if it passes every check.</p>
      <div className="scroll" role="region" aria-label="Model comparison" tabIndex={0}>
        <table>
          <thead><tr><th scope="col">Model</th><th scope="col" className="num">Accuracy</th><th scope="col">95% interval</th><th scope="col" className="num">Macro F1</th><th scope="col" className="num">Calibration error</th><th scope="col">Gate</th></tr></thead>
          <tbody>
            {data.models.map((x) => (
              <tr key={x.version} className={x.version === data.champion ? 'is-champion' : ''}>
                <th scope="row">{x.kind}{x.version === data.champion && <i className="pill pill--answered">in use</i>}</th>
                <td className="num">{pct(x.metrics.accuracy, 1)}</td>
                <td>{pct(x.metrics.interval[0])} to {pct(x.metrics.interval[1])}</td>
                <td className="num">{x.metrics.macro_f1.toFixed(2)}</td>
                <td className="num">{pct(x.metrics.ece, 1)}</td>
                <td><ul className="checks">{Object.entries(x.gate.checks).map(([k, v]) => <li key={k} className={v ? 'ok' : 'bad'}><span aria-hidden="true">{CHECK[v]}</span> <span className="sr">{v ? 'Passed: ' : 'Failed: '}</span>{k}</li>)}</ul></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>When it says 80%, is it right 80% of the time?</h2>
      <p>Dots on the line mean honest confidence. Larger dots hold more messages.</p>
      <Reliability models={data.models} />

      <h2>Where it goes wrong</h2>
      <div className="seg" role="group" aria-label="Model">
        {data.models.map((x) => <button key={x.version} type="button" aria-pressed={x.version === version} onClick={() => setVersion(x.version)}>{x.version}</button>)}
      </div>
      <Confusion m={m} labels={labels} />
      <details>
        <summary>The {data.mistakes.length} messages {data.champion} gets wrong</summary>
        <ul className="mistakes">{data.mistakes.map((x) => <li key={x.text}><q>{x.text}</q> <span>{x.truth}, read as {x.said}</span></li>)}</ul>
      </details>

      <h2>Would we notice if the mail changed?</h2>
      <p>No labels are needed. The stability index compares the mix of topics and confidence with what the model was checked on. Above 0.25 is an alarm.</p>
      <div className="scroll" role="region" aria-label="Drift checks" tabIndex={0}>
        <table>
          <thead><tr><th scope="col">Mail</th><th scope="col" className="num">Topics</th><th scope="col" className="num">Confidence</th><th scope="col" className="num">New words</th><th scope="col">State</th></tr></thead>
          <tbody>{data.drift.map((d) => <tr key={d.name}><th scope="row">{d.name}</th><td className="num">{d.intent_psi.toFixed(2)}</td><td className="num">{d.confidence_psi.toFixed(2)}</td><td className="num">{pct(d.new_words)}</td><td><i className={`pill pill--${d.state}`}>{d.state}</i></td></tr>)}</tbody>
        </table>
      </div>
    </div>
  )
}
