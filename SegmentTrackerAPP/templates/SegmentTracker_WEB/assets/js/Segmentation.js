

function Segmentation(){
    makeRequest("doSegmentation/")
    alert("Ready")
    //makeRequest("load_segmented_images/")
    window.location.replace("resultado_procesar_Imagenes.html");
}