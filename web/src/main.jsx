import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import '@fontsource-variable/instrument-sans'
import '@fontsource-variable/jetbrains-mono'
import './styles/main.scss'
import App from './App.jsx'
import recorded from './recorded.json'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter basename={import.meta.env.BASE_URL.replace(/\/$/, '')}><App data={recorded} /></BrowserRouter>
  </StrictMode>,
)
