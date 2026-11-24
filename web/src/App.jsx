import { NavLink, Route, Routes } from 'react-router-dom'
import Inbox from './pages/Inbox.jsx'
import Models from './pages/Models.jsx'
import Search from './pages/Search.jsx'

export default function App({ data }) {
  return (
    <>
      <a className="skip" href="#main">Skip to content</a>
      <header className="top">
        <p className="top__brand">Open Desk</p>
        <nav aria-label="Main">
          <NavLink to="/" end>Inbox</NavLink>
          <NavLink to="/models">Models</NavLink>
          <NavLink to="/search">Search</NavLink>
        </nav>
        <p className="top__note">Recorded run. Nothing is sent. <a href="https://github.com/Atharv1414/ai-support-desk">Code</a></p>
      </header>
      <main id="main">
        <Routes>
          <Route path="/models" element={<Models data={data} />} />
          <Route path="/search" element={<Search data={data} />} />
          <Route path="*" element={<Inbox data={data} />} />
        </Routes>
      </main>
    </>
  )
}
