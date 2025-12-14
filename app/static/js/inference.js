document.getElementById("backIcon").addEventListener("click", () => history.back())

let infoDialog = document.getElementById("infoDialog")

document.getElementById("infoIcon").addEventListener("click", () => infoDialog.open = true)

let plots = document.getElementById("plotsDiv")
let predictedIcon = document.getElementById("predictedIcon")
let trueIcon = document.getElementById("trueIcon")
let errorIcon = document.getElementById("errorIcon")
let residualsIcon = document.getElementById("residualsIcon")

function mdColorToPlotly(color) {
    return "rgb(" + color + ")"
}

let style = getComputedStyle(document.documentElement)
let backgroundColor = mdColorToPlotly(style.getPropertyValue("--mdui-color-background"))
let primaryColor = mdColorToPlotly(style.getPropertyValue("--mdui-color-primary"))
let secondaryColor = mdColorToPlotly(style.getPropertyValue("--mdui-color-inverse-primary"))
let onSurfaceColor = mdColorToPlotly(style.getPropertyValue("--mdui-color-on-surface"))
let gridcolor = mdColorToPlotly(style.getPropertyPriority("--mdui-color-on-surface-variant"))

let data = JSON.parse(sessionStorage.getItem("data"))
console.log(data)

var trace1 = {
    x: data.points.x,
    y: data.points.y,
    mode: 'markers',
    type: 'scatter',
    marker: { color: data.predicted.Ux },
    colorscale: "balance"
};


var trace2 = {
    x: data.points.x,
    y: data.points.y,
    xaxis: 'x2',
    yaxis: 'y',
    mode: 'markers',
    type: 'scatter',
    marker: { color: data.predicted.Uy },
};


var trace3 = {
    x: data.points.x,
    y: data.points.y,
    xaxis: 'x',
    yaxis: 'y3',
    mode: 'markers',
    type: 'scatter',
    marker: { color: data.predicted.p },
};


var trace4 = {
    x: data.points.x,
    y: data.points.y,
    xaxis: 'x2',
    yaxis: 'y3',
    mode: 'markers',
    type: 'scatter',
    marker: { color: data.predicted.U },
};

let traces = [trace1, trace2, trace3, trace4];
let xaxis = {
    title: { text: "x" },
    range: [-0.4, 0.6],
    gridcolor: gridcolor,
    zeroline: false,
    dtick: 0.1
}
let yaxis = {
    title: { text: "y" },
    range: [-0.3, 0.3],
    gridcolor: gridcolor,
    zeroline: false,
    dtick: 0.1
}

let layout = {
    "xaxis": xaxis,
    "yaxis": yaxis,
    "xaxis2": xaxis,
    "yaxis3": yaxis,
    grid: {
        rows: 2,
        columns: 2,
        subplots: [['xy3', 'x3y4'], ['xy', 'x2y']],
        roworder: 'bottom to top'
    },
    aspectmode: "equal",
    paper_bgcolor: backgroundColor,
    plot_bgcolor: backgroundColor,
    font: {
        color: onSurfaceColor
    }
}

Plotly.newPlot(plots, traces, layout)

predictedIcon.addEventListener("focus", () => {
    Plotly.restyle(plots, { "marker.color": [data.predicted.Ux, data.predicted.Uy, data.predicted.p, data.predicted.U] })
})

trueIcon.addEventListener("focus", () => {
    Plotly.restyle(plots, { "marker.color": [data.target.Ux, data.target.Uy, data.target.p, data.target.U] })
})

errorIcon.addEventListener("focus", () => {
    Plotly.restyle(plots, { "marker.color": [data.error.Ux, data.error.Uy, data.error.p, data.error.U] })
})

residualsIcon.addEventListener("focus", () => {
    Plotly.restyle(plots, { "marker.color": [data.residuals.Momentumx, data.residuals.Momentumx, data.residuals.div, data.residuals.Momentum] })
})