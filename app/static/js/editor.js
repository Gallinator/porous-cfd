let uuid = self.crypto.randomUUID();

let myDiv = document.getElementById("myDiv")

function mdColorToPlotly(color) {
  return "rgb(" + color + ")"
}

const example = document.getElementById("baseContainer");
const navigationDrawer = example.querySelector("mdui-navigation-drawer");
const menuIcon = document.getElementById("menuIcon");
const acceptIcon = document.getElementById("acceptIcon")
const splineDegreeSlider = document.getElementById("splineDegreeSlider")
const splinePointsSlider = document.getElementById("splinePointsSlider")
const modelSelector = document.getElementById("modelSelector")


let style = getComputedStyle(document.documentElement)
let backgroundColor = mdColorToPlotly(style.getPropertyValue("--mdui-color-background"))
let primaryColor = mdColorToPlotly(style.getPropertyValue("--mdui-color-primary"))
let secondaryColor = mdColorToPlotly(style.getPropertyValue("--mdui-color-inverse-primary"))
let onSurfaceColor = mdColorToPlotly(style.getPropertyValue("--mdui-color-on-surface"))
let gridcolor = mdColorToPlotly(style.getPropertyPriority("--mdui-color-on-surface-variant"))

let curve = new BSpline(splineDegreeSlider.value, new Array(
  new Point(- 0.1, -0.1),
  new Point(-0.1, 0.1),
  new Point(0.1, 0.1),
  new Point(0.1, -0.1),
  new Point(-0.1, -0.1),
  new Point(- 0.1, 0.1)), true)
let nPoints = splinePointsSlider.value

let curveData = curve.getPlotlyCurve(nPoints)
var curveTrace = {
  x: curveData.x,
  y: curveData.y,
  mode: 'lines+markers',
  line: { color: primaryColor }
};


let controlPoints = curve.getPlotlyControlPoints()
var controlTrace = {
  x: controlPoints.x,
  y: controlPoints.y,
  mode: 'markers',
  marker: {
    color: secondaryColor,
    size: 20
  }
}

var layout = {
  xaxis: {
    title: { text: "x" },
    range: [-0.4, 0.6],
    gridcolor: gridcolor,
    zerolinecolor: onSurfaceColor,
    dtick: 0.1
  },
  yaxis: {
    title: { text: "y" },
    range: [-0.3, 0.3],
    gridcolor: gridcolor,
    zerolinecolor: onSurfaceColor,
    dtick: 0.1
  },
  aspectmode: "equal",
  showlegend: false,
  dragmode: false,
  paper_bgcolor: backgroundColor,
  plot_bgcolor: backgroundColor,
  font: {
    color: onSurfaceColor
  }
}

let config = { staticPlot: true }

Plotly.newPlot(myDiv, [curveTrace, controlTrace], layout, config)

let curveEditor = new CurveEditor(myDiv, curve, 0.005)

curveEditor.onupdate = function () {
  let curvePoints = curve.getPlotlyCurve(nPoints)
  let curveData = myDiv.data[0]
  curveData.x = curvePoints.x
  curveData.y = curvePoints.y

  let controlData = myDiv.data[1]
  let controlPoints = curve.getPlotlyControlPoints()
  controlData.x = controlPoints.x
  controlData.y = controlPoints.y
  Plotly.redraw('myDiv')
}

menuIcon.addEventListener("click", () => {
  navigationDrawer.open = true
  curveEditor.enabled = false
});

navigationDrawer.addEventListener("close", () => curveEditor.enabled = true)

acceptIcon.addEventListener("click", () => {

})

splinePointsSlider.addEventListener("input", () => {
  nPoints = splinePointsSlider.value
  console.log(curveEditor.curve)
  curveEditor.onupdate()
})

splineDegreeSlider.addEventListener("change", () => {
  let cPoints = curve.controlPoints
  cPoints.splice(-curve.degree)

  let newDegree = splineDegreeSlider.value
  for (let i = 0; i < newDegree; i++)
    cPoints.push(JSON.parse(JSON.stringify(cPoints[i])))

  curve = new BSpline(newDegree, cPoints, curve.closed)
  curveEditor.curve = curve
  curveEditor.onupdate()
})

acceptIcon.addEventListener("click", async () => {
  let body = {
    "uuid":uuid,
    model: modelSelector.value.toLowerCase(),
    points: curve.getPlotlyCurve(nPoints)
  }

  let response = await fetch("predict",
    {
      method: "POST",
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(body)
    })

  let data = await response.json()

  sessionStorage.setItem("data", JSON.stringify(data))

  window.open("inference/index.html", "_self")
})