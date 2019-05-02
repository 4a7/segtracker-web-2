var xmlHttpRequest = new XMLHttpRequest();

function makeRequest(query){
    query = "/SegmentTrackerAPP/" + query 
    //alert("El query: " + query);
    xmlHttpRequest.open( "GET", query, false ); // false for synchronous request
	xmlHttpRequest.send();
}

function alertar(mensaje){
    alert(mensaje);
}

/*
Esto es para trabajar con respuestas asicronicas
function example(){
    makeRequest(query);	
    xmlHttpRequest.onreadystatechange = function() {
    if(xmlHttpRequest.readyState == XMLHttpRequest.DONE && xmlHttpRequest.status == 200) {
            alert(xmlHttpRequest.responseText);
        }
    }
}
*/
