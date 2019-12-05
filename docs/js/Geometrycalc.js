

function rectangleArea () {
	

	var rectangleBase = document.getElementById("rectangleBase").value;
	var	rectangleHeight = document.getElementById("rectangleHeight").value;
	// body...
	 
	document.getElementById("RectangleArea").innerHTML = rectangleBase * rectangleHeight;
}
function triangleArea(){

	var triangleBase = document.getElementById("triangleBase").value;
	var	triangleHeight = document.getElementById("triangleHeight").value;
	// body...
	 
	document.getElementById("TriangleArea").innerHTML = 0.5 * triangleBase * triangleHeight;

}