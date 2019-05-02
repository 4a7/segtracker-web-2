class SelectFile {
    constructor() {
        this.getNodes();
        this.addEventListeners();
    }
    
    getNodes() {
        this.addFileBtn = document.querySelector(".add-file");
        this.selectedFile = document.querySelector(".selectedfile");
    }
    
    addEventListeners() {
        this.addFileBtn.addEventListener("click", this.createFile.bind(this));
    }
    
    createFile() {
        const clonedFile = this.selectedFile.cloneNode(true);
        clonedFile.classList.add("selectedfile-show");
        
        const label = clonedFile.querySelector("span");
        label.textContent = this.getRandomLabel();
        
        this.selectedFile.parentNode.insertBefore(clonedFile, this.selectedFile);
        
        this.changeBtn();
    }
    
    getRandomLabel() {
        const labelType = ["verslag", "evaluatie", "reflectie", "onderzoek", "testplan"];
        const labelName = ["Grover", "Emmy", "Victor", "Kelly", "Jazmin"];
        const labelNum = ["19344", "72464", "63782", "85300", "27542"];
        const labelExt = ["pdf", "docx", "pptx", "xlsx", "txt"];
        
        return `${this.getRandomItem(labelType)}_${this.getRandomItem(labelName)}_${this.getRandomItem(labelNum)}.${this.getRandomItem(labelExt)}`;
    }

    getRandomItem(arr) {
        return arr[Math.floor(Math.random() * arr.length)];
    }
    
    changeBtn() {
        this.addFileBtn.classList.remove("w-100");
        this.addFileBtn.querySelector("span").classList.add("add-file-hide");
    }
}

const selectFile = new SelectFile();