import { createRoot } from 'react-dom/client'
import './theme.css'
import App from './App.jsx'

// No <React.StrictMode> here: it double-invokes effects in dev, and
// MapLibre GL shares a global Web Worker pool across Map instances —
// StrictMode's discarded first mount tears that pool down mid-load,
// silently breaking the second (real) map's tile fetching.
createRoot(document.getElementById('root')).render(<App />)
