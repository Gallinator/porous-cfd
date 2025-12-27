
BSpline.prototype.getPlotlyCurve = function (nPoints) {
    let x = [], y = []
    let delta = 1 / nPoints
    let domain = { l: this.knots[this.degree], h: this.knots[this.knots.length - 1 - this.degree] }

    let pL = this.getValue(domain.l)
    x.push(pL.x)
    y.push(pL.y)

    for (let i = 0; i <= nPoints; i++) {
        let t = i * delta
        if (t <= domain.l || t >= domain.h)
            continue
        let p = this.getValue(t)
        x.push(p.x)
        y.push(p.y)
    }

    x.push(pL.x)
    y.push(pL.y)

    return { "x": x, "y": y }
};

BSpline.prototype.getPlotlyControlPoints = function () {
    let x = [], y = []
    for (let i = 0; i < this.controlPoints.length - this.degree; i++) {
        x.push(this.controlPoints[i].x)
        y.push(this.controlPoints[i].y)
    }
    return { "x": x, "y": y }
}

function drawControlPoints(canvas) {
    let ctx = canvas.getContext("2d")
    for (let i = 1; i < this.controlPoints.length - 1; i++) {
        ctx.beginPath()
        let center = this.controlPoints[i]
        ctx.arc(center.x, center.y, 10, 0, 2 * Math.PI)
        ctx.strokeStyle = (i == 1 || i == this.controlPoints.length - 2) ? "blue" : "red";
        ctx.lineWidth = 2
        ctx.stroke()
    }
};

function draw(canvas, nPoints) {
    let ctx = canvas.getContext("2d")
    ctx.lineWidth = 5
    ctx.beginPath()
    let delta = 1 / nPoints
    let curP = this.getValue(0)
    ctx.moveTo(curP.x, curP.y)
    for (let i = 1; i <= nPoints; i++) {
        curP = this.getValue(i * delta)
        ctx.lineTo(curP.x, curP.y)
        ctx.strokeStyle = "black"
        ctx.lineCap = "round"
        ctx.stroke()
    }
    if (this.closed) {
        ctx.fillStyle = 'grey'
        ctx.fill()
    }
};


function denormalize(val, range) {
    return val * (range[1] - range[0]) + range[0]
}

function clamp(v, l, h) {
    return Math.min(Math.max(v, l), h)
}

function getPlotPixelSize(div) {
    let clipRect = div._fullLayout._plots.xy.clipRect[0][0]
    return { "x": clipRect.width.baseVal.value, "y": clipRect.height.baseVal.value }
}

function getLayerPos(div, eventData) {
    let rect = div.getBoundingClientRect()
    let margins = div._fullLayout.margin
    return {
        "x": eventData.clientX - rect.x - margins.l + window.scrollX,
        "y": eventData.clientY - rect.y - margins.t + window.screenY
    }
}

function getPlotPos(div, eventData) {
    let plotPos = getLayerPos(div, eventData)
    let plotSize = getPlotPixelSize(div)

    let x = plotPos.x / plotSize.x
    x = clamp(x, 0, 1)
    let y = 1 - plotPos.y / plotSize.y
    y = clamp(y, 0, 1)

    return {
        "x": denormalize(x, div._fullLayout.xaxis.range),
        "y": denormalize(y, div._fullLayout.yaxis.range)
    }
}

function isInsidePlot(div, eventData) {
    let docPos = {
        "x": div.getBoundingClientRect().x + window.scrollX,
        "y": div.getBoundingClientRect().y + window.scrollY
    }
    let margins = div._fullLayout.margin
    let localX = eventData.clientX - docPos.x
    let localY = eventData.clientY - docPos.y
    let plotSize = getPlotPixelSize(div)
    return localX > margins.l && localX < margins.l + plotSize.x && localY > margins.t && localY < margins.t + plotSize.y
}

class CurveEditor {
    constructor(div, curve, selectThreshold) {
        this.div = div
        this.curve = curve
        this.selectThreshold = selectThreshold

        this.enabled = true
        this.moving = false
        this.movingId = false

        this.handler = { onupdate: () => { } }

        this.div.ownerDocument.addEventListener("mousedown", (evt) => {
            if (!isInsidePlot(this.div, evt) || !this.enabled) return
            let clickedPoint = getPlotPos(div, evt);

            this.curve.controlPoints.forEach((p, i) => {
                if (this.isWithinThreshold(clickedPoint, p) && this.isSelectable(i)) {
                    this.moving = true;
                    this.movingId = i;
                }
            })
        });

        this.div.ownerDocument.addEventListener("mousemove", (evt) => {
            let found = false;
            let mousePoint = getPlotPos(this.div, evt)

            this.curve.controlPoints.forEach((p, i) => {
                if (this.isWithinThreshold(mousePoint, p) && this.isSelectable(i))
                    found = found || true;
            });

            this.div.style.cursor = found ? "pointer" : "default";

            if (!this.moving)
                return

            this.curve.moveControlPoint(this.movingId, mousePoint.x, mousePoint.y)

            this.onupdate();
        });

        this.div.ownerDocument.addEventListener("mouseup", (evt) => {
            if (!this.moving) return;
            this.moving = false;
        });

        this.div.ownerDocument.addEventListener('contextmenu', (event) => {
            if (!isInsidePlot(this.div, event) || !this.enabled) return

            event.preventDefault();

            let clickedPoint = getPlotPos(this.div, event)
            this.curve.addControlPoint(new Point(clickedPoint.x, clickedPoint.y));
            this.onupdate();
        });
    }

    getDistance(p1, p2) {
        return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2))
    }

    isWithinThreshold(pointer, tgt) {
        return Math.abs(pointer.x - tgt.x) < this.selectThreshold && Math.abs(pointer.y - tgt.y) < this.selectThreshold
    }

    isSelectable(i) {
        return i < this.curve.controlPoints.length - this.curve.degree
    }
}