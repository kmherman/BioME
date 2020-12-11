

function  changePics(v) {
    var target = document.getElementById("pics");
    switch(v){
        case "KNN":
            target.innerHTML= "<img height=300 width=450 src='github_page_file/image/knn.png'>";
            break;
        case "Logistic":
            target.innerHTML= "<img height=320 width=450 src='github_page_file/image/logisitic.jpeg'>";
            break;
        case "Ridge":
            target.innerHTML= "<img height=320 width=450 src='github_page_file/image/ridge.jpg'>";
            break;
        case "Tree":
            target.innerHTML= "<img height=320 width=450 src='github_page_file/image/decision-trees.jpg'>";
            break;
        case "Random Forest":
            target.innerHTML= "<img height=320 width=450 src='github_page_file/image/Random_Forest.png'>";
            break;
        case "Naive Bayes":
            target.innerHTML= "<img height=350 width=550 src='github_page_file/image/naive bayes.png'>";
            break;
        case "NNs":
            target.innerHTML= "<img height=300 width=420 src=github_page_file/image/'NNs.png'>";
            break;
    }
  }