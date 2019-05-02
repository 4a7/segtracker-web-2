
function verify_user(){
    useremail = document.getElementById("user_email").value;
    userpassword = document.getElementById("user_password").value;

    //alert(useremail+", "+userpassword)

    var email_regex = /^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,4})+$/
    var password_regex = /^([A-Za-z0-9]{6,15})+$/;
    var ok = true;
    
    if(!email_regex.test(useremail)){
        alert("Por favor ingrese un correo electrónico válido.")   
        ok = false
        return     
    }

    if(!password_regex.test(userpassword)){
        alert("La contraseña debe contener únicamente letras y números con una extensión de 6 a 15 caracteres.")
        ok = false
        return
    }

    if(ok == true){
        makeRequest("signing/?email="+useremail+"&password="+userpassword)
        //alert("CONTRASEÑA: "+xmlHttpRequest.responseText)
        //alert("LISTO")
        var x = xmlHttpRequest.responseText
        var key = "segtrackerpasskey";
        var decrypted = x //CryptoJS.AES.decrypt(x, key)

        if(userpassword == x){
            //alert("TODO BIEN")
            window.location.replace("nueva_segmentacion.html")
        }else{
            alert("El correo o la contraseña es incorrecta.")
        }
        //xmlHttpRequest.onreadystatechange = function() {
        //    if(xmlHttpRequest.readyState == XMLHttpRequest.DONE && xmlHttpRequest.status == 200) {
        //            alert(xmlHttpRequest.responseText);
        //    }
        //}
    //var decrypted = CryptoJS.AES.decrypt(encrypted, key);
    //alert(pass);
    //alert(encrypted);
    //alert(decrypted);
    //alert(decrypted.toString(CryptoJS.enc.Utf8));
        
    } else{
        alert("No se pudo iniciar sesión.")
    }

}

