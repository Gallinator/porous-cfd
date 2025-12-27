document.getElementById("backIcon").addEventListener("click", () => history.back())

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

let storedData = JSON.parse(sessionStorage.getItem("data"))
let rawData = storedData.raw_data
let gridData = storedData.grid_data

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

    let layoutUpdate = {
        "title.text": title
    }

    Plotly.update(plotDiv, contourUpdate, layoutUpdate, [1])

    let equalAspectPlotSize = getEqualAspectSize(plotDiv, 1 / 0.6)
    Plotly.update(plotDiv, { "hovertemplate": "%{customdata:.3f}" + " " + unitText },
        { height: equalAspectPlotSize.y, autosize: false }, [0])
    // This has tp be manually set because using restyle does not work
    plotDiv._fullData[0].customdata = rawData
}

createPlot("$U_x$", tlPlot, rawData.points, rawData.predicted.Ux, gridData.points, gridData.predicted.Ux, "m \\ s")
createPlot("$U_y$", trPlot, rawData.points, rawData.predicted.Uy, gridData.points, gridData.predicted.Uy, "m \\ s")
createPlot("$p$", blPlot, rawData.points, rawData.predicted.p, gridData.points, gridData.predicted.p, "m² \\ s²")
createPlot("$U_x$", brPlot, rawData.points, rawData.predicted.U, gridData.points, gridData.predicted.U, "m \\ s")

predictedIcon.addEventListener("focus", () => {
    updatePlot(tlPlot, rawData.predicted.Ux, gridData.predicted.Ux, "m \\ s", "$U_x$")
    updatePlot(trPlot, rawData.predicted.Uy, gridData.predicted.Uy, "m \\ s", "$U_y$")
    updatePlot(blPlot, rawData.predicted.p, gridData.predicted.p, "m² \\ s²", "$p$")
    updatePlot(brPlot, rawData.predicted.U, gridData.predicted.U, "m \\ s", "$U$")
})

trueIcon.addEventListener("focus", () => {
    updatePlot(tlPlot, rawData.target.Ux, gridData.target.Ux, "m \\ s", "$U_x$")
    updatePlot(trPlot, rawData.target.Uy, gridData.target.Uy, "m \\ s", "$U_y$")
    updatePlot(blPlot, rawData.target.p, gridData.target.p, "m² \\ s²", "$p$")
    updatePlot(brPlot, rawData.target.U, gridData.target.U, "m \\ s", "$U$")
})

errorIcon.addEventListener("focus", () => {
    updatePlot(tlPlot, rawData.error.Ux, gridData.error.Ux, "m \\ s", "$U_x$")
    updatePlot(trPlot, rawData.error.Uy, gridData.error.Uy, "m \\ s", "$U_y$")
    updatePlot(blPlot, rawData.error.p, gridData.error.p, "m² \\ s²", "$p$")
    updatePlot(brPlot, rawData.error.U, gridData.error.U, "m \\ s", "$U$")
})

residualsIcon.addEventListener("focus", () => {
    updatePlot(tlPlot, rawData.residuals.Momentumx, gridData.residuals.Momentumx, "", "$Momentum_x$")
    updatePlot(trPlot, rawData.residuals.Momentumy, gridData.residuals.Momentumy, "", "$Momentum_y$")
    updatePlot(blPlot, rawData.residuals.div, gridData.residuals.div, "", "$Divergence$")
    updatePlot(brPlot, rawData.residuals.Momentum, gridData.residuals.Momentum, "", "$Momentum$")
})