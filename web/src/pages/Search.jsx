import { useEffect } from 'react'
import { pct } from '../lib.js'

const NAMES = { bm25: 'Words (BM25)', semantic: 'Gist (LSA)', hybrid: 'Both, fused' }

export default function Search({ data }) {
  const r = data.retrieval
  useEffect(() => {
    document.title = 'Search – Open Desk'
  }, [])
  return (
    <div className="page">
      <h1>Can it find the right answer, and stay quiet when there is none?</h1>
      <p className="lead">Questions were split by section. One half set the cut-off for refusing, the other half is reported here.</p>
      <div className="scroll" role="region" aria-label="Search quality" tabIndex={0}>
        <table>
          <thead><tr><th scope="col">Method</th><th scope="col" className="num">Right section first</th><th scope="col" className="num">In the top three</th><th scope="col" className="num">Mean reciprocal rank</th></tr></thead>
          <tbody>{Object.keys(NAMES).map((k) => <tr key={k}><th scope="row">{NAMES[k]}</th><td className="num">{pct(r.test[k]['recall@1'])}</td><td className="num">{pct(r.test[k]['recall@3'])}</td><td className="num">{r.test[k].mrr.toFixed(2)}</td></tr>)}</tbody>
        </table>
      </div>
      <dl className="facts">
        <div><dt>Refuses questions the help centre cannot answer</dt><dd>{pct(r.test.refuses_unanswerable)}</dd></div>
        <div><dt>Refuses questions it could have answered</dt><dd>{pct(r.test.refuses_answerable)}</dd></div>
      </dl>
      <p>A refusal goes to a person, so a wrong refusal costs time and a wrong answer costs trust. The cut-off leans towards refusing.</p>
      <h2>Help centre</h2>
      <ul className="sections">
        {Object.entries(data.sections).map(([id, s]) => <li key={id}><h3>{s.title}</h3><p>{s.text}</p></li>)}
      </ul>
    </div>
  )
}
