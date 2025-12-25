document.getElementById("backIcon").addEventListener("click", () => history.back())

let infoDialog = document.getElementById("infoDialog")

document.getElementById("infoIcon").addEventListener("click", () => infoDialog.open = true)

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
    title: { text: "$x$" },
    range: [-0.4, 0.6],
    minallowed: -0.4,
    maxallowed: 0.6,
    "gridcolor": gridcolor,
    zeroline: false,
    dtick: 0.1
}
let yaxis = {
    title: { text: "$y$" },
    range: [-0.3, 0.3],
    minallowed: -0.3,
    maxallowed: 0.3,
    "gridcolor": gridcolor,
    zeroline: false,
    dtick: 0.1
}

let tlPlot = document.getElementById("tlPlot")
let trPlot = document.getElementById("trPlot")
let blPlot = document.getElementById("blPlot")
let brPlot = document.getElementById("brPlot")

function getEqualAspectSize(plotDiv, aspect) {
    let plotPixSize = getPlotPixelSize(plotDiv)
    let margins = plotDiv._fullLayout.margin
    plotPixSize.y = plotPixSize.x * 1 / aspect

    plotPixSize.x += margins.l + margins.r
    plotPixSize.y += margins.t + margins.b
    return plotPixSize
}

function createPlot(title, plotDiv, rawPoints, rawData, gridPoints, gridField, unitText) {
    let scatterTrace = {
        x: rawPoints.x,
        y: rawPoints.y,
        mode: 'markers',
        type: 'scatter',
        marker: { color: "black" },
        colorscale: "balance",
        name: "",
        customdata: rawData,
        hovertemplate: "%{customdata:.3f}" + " " + unitText
    }

    let contourTrace = {
        x: gridPoints.x,
        y: gridPoints.y,
        z: gridField,
        type: 'contour',
        colorscale: "Portland",
        contours: { coloring: 'heatmap' },
        ncontours: 20,
        line: { width: 0 },
        colorbar: {
            title: { text: unitText, color: onSurfaceColor },
            thickness: 0.025,
            thicknessmode: "fraction"
        },
        name: "",
        hovertemplate: false,
        hoverinfo: "skip"
    }

    let layout = {
        "title": { text: title, font: { color: onSurfaceColor } },
        "xaxis": xaxis,
        "yaxis": yaxis,
        paper_bgcolor: backgroundColor,
        plot_bgcolor: backgroundColor,
        font: { color: onSurfaceColor },
        showlegend: false
    }

    let config = {
        responsive: true,
        displaylogo: false
    }

    Plotly.newPlot(plotDiv, [scatterTrace, contourTrace], layout, config)

    // Update the plot to have the correct aspect ratio after automatically calculating the margins
    let equalAspectPlotSize = getEqualAspectSize(plotDiv, 1 / 0.6)
    Plotly.relayout(plotDiv, { height: equalAspectPlotSize.y, autosize: false })
}

function updatePlot(plotDiv, rawData, gridData, unitText, title) {
    let contourUpdate = {
        "z": [gridData],
        "colorbar.title.text": unitText
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