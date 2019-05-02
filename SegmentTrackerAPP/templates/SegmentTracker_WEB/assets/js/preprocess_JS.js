
var selected_images = []

function preprocess(element_id){
    //selected_images = []
    //alert(element_id.title)
    //alert(element_id.id)

    if(element_id.title === "unchecked"){
        selected_images.push(element_id.id)
        document.getElementById(element_id.id).setAttribute("style", "color: rgb(60, 165, 53);");
        element_id.title =  "checked"
    }else if(element_id.title === "checked"){
        selected_images.splice( selected_images.indexOf(element_id.id), 1 );
        document.getElementById(element_id.id).setAttribute("style", "color: rgb(98, 98, 98);");
        element_id.title =  "unchecked"
    }
}

function doPreprocess(){
    //alert("ESTO ES EL PRERPOCESAMIENTO")
    if(selected_images.length > 0){
        for(i = 0 ; i<selected_images.length ; i++){
            makeRequest("do_denoising/?image="+selected_images[i])
            //alert(selected_images[i])
        }
        alert("Las imágenes han sido preprocesadas.")
    }
    window.location.replace("loading.html")
}

function deleteImage(obj){
    var x = confirm("¿Está seguro de que desea eliminar la imagen seleccionada?");
    if(x == true){
        makeRequest("remove_image/?image="+obj.id)
        document.getElementById("img_"+obj.id).remove();
    }
}

/*
function Segmentation(){
    doPreprocess();
    window.location.replace("loading.html")
}
*/
