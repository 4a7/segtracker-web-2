function checkUs(){
    makeRequest("loggedUser/");
    var result = xmlHttpRequest.responseText
    if(result != "ok"){
        alert("Debe iniciar sesión para acceder a esta funcionalidad")
        window.location.replace("/SegmentTrackerAPP")
    }
}

function logout(){
    makeRequest("logoutUser/");
    //var result = xmlHttpRequest.responseText
    //alert(result)
    //window.location.replace("index.html")
}
