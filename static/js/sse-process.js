const appContainer = document.querySelector('div.app')
const outside = document.querySelector('div.outside')
const bar = document.querySelector('div.inside')

const sse = new EventSource('/api/processing', { withCredentials: true})

sse.onmessage = e => {
    // progress message
    console.log(`${e.data}`)
    const parts = e.data.split('/')
    const current = parseInt(parts[0])
    const total = parseInt(parts[1])

    // update UI
    const baseWidth = 300 //parseInt(outside.style.width)
    const barWidth = Math.floor( (current / total) * baseWidth )
    bar.style.width = `${barWidth}px`

    // redirect if progress ended
    if (current == total){
        sse.close()
        window.location.pathname = '/result_file'
    }        
}
