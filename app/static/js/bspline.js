class Point {
    constructor(x, y) {
        this.x = x
        this.y = y
    };
}
function multNumber(v1, v2) {
    return new Point(v1.x * v2, v1.y * v2)
};
function sumPoint(v1, v2) {
    return new Point(v1.x + v2.x, v1.y + v2.y)
}

class BSpline {
    constructor(degree, controlPoints, closed) {
        this.degree = degree
        this.controlPoints = controlPoints
        this.knots = this.createKnots()
        this.closed = closed

    };

    updateOverlappingControlPoints() {
        if (!this.closed)
            return

        for (let i = 0; i < this.degree; i++) {
            let anchorPoint = this.controlPoints[i]
            let overlapId = this.controlPoints.length - (this.degree - i)
            this.controlPoints[overlapId].x = anchorPoint.x
            this.controlPoints[overlapId].y = anchorPoint.y
        }
    }

    addControlPoint(p, insertId) {
        this.controlPoints.splice(insertId, 0, p);
        this.updateOverlappingControlPoints()
        this.knots = this.createKnots()
    };

    deleteControlPoint(id) {
        this.controlPoints.splice(id, 1)
        this.updateOverlappingControlPoints()
        this.knots = this.createKnots()
    }

    moveControlPoint(i, posX, posY) {
        this.controlPoints[i].x = posX
        this.controlPoints[i].y = posY
        if (this.closed && i < this.degree) {
            let idx = this.controlPoints.length - this.degree + i
            this.controlPoints[idx].x = posX
            this.controlPoints[idx].y = posY
        }
    };

    /**
     * @param t The position to evaluate the at, in range 0..1
     * @return The point on spline at [t]
     */
    getValue(t) {
        let range = this.getKnotRangeIndex(t)
        var temporaryPoints = Array.from({ length: this.degree + 1 }, (_, j) =>
            JSON.parse(JSON.stringify(this.controlPoints[j + range - this.degree])))

        for (let r = 1; r <= this.degree; r++) {
            for (let j = this.degree; j >= r; j--) {
                let alpha = (t - this.knots[j + range - this.degree]) / (this.knots[j + 1 + range - r] - this.knots[j + range - this.degree])
                temporaryPoints[j] = sumPoint(multNumber(temporaryPoints[j - 1], (1 - alpha)), (multNumber(temporaryPoints[j], (alpha))))
            }
        }
        return temporaryPoints[this.degree]
    };

    /**
     * Creates equally spaced knots which allows the b spline to be clamped at the start and end point.
     */
    createKnots() {
        let knots = new Array(this.controlPoints.length + this.degree + 1)
        let knotsLastId = knots.length - 1
        let knotsValuesRange = knotsLastId - 2 * this.degree

        for (let i = 0; i < knots.length; i++) {

            knots[i] = (i) / knotsLastId
        }
        return knots
    };

    /**
     * Finds the points between which [t] lies as a range index.
     */
    getKnotRangeIndex(t) {
        let knotsLastId = this.knots.length - 1
        if (t <= this.knots[0] && this.knots[0] == this.knots[this.degree])
            return this.degree
        if (t >= this.knots[knotsLastId] && this.knots[knotsLastId] == this.knots[knotsLastId - this.degree])
            return knotsLastId - this.degree - 1

        for (let i = 0; i < knotsLastId; i++)
            if (t >= this.knots[i] && t < this.knots[i + 1])
                return i

        return 0
    };
}