
function DataSet_Session(){
    var detail =  document.getElementById("detail").value
    //alert(detail)

    if(checkdetail(detail) == true){
        makeRequest("DataSetSession/?detail="+detail)
        window.location.replace("Cargar_imagenes.html")
    }
    else{
    var present = xmlHttpRequest.responseText
    //alert("Data: ", present)
    }
    //window.location.replace("Cargar_imagenes.html")
    
}

function checkdetail(detail){
    var detail_regex = /^[A-ZÀ-ÿa-z0-9,.\/]+[A-ZÀ-ÿa-z0-9,.\/\n ]*$/;
    if(!detail_regex.test(detail)){
        alert("Por favor ingrese un detalle válido.")
        return false
    }
    return true
}
