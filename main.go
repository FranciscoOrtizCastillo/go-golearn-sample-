package main

import (
	"fmt"
	"log"
	"os"

	"github.com/go-gota/gota/dataframe"

	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/knn"
)

func Manipulation() {
	irisCsv, err := os.Open("./iris_headers.csv")
	if err != nil {
		log.Fatal(err)
	}

	df := dataframe.ReadCSV(irisCsv)
	fmt.Println(df)

	// Data Manipulation in Go
	// --------------------------------------------------

	// Subsetting
	fmt.Println("-----------------------------------------------------")
	fmt.Println("Subsetting")
	fmt.Println("-----------------------------------------------------")

	head := df.Subset([]int{0, 3})
	fmt.Println(head)

	// Filtering
	fmt.Println("-----------------------------------------------------")
	fmt.Println("Filtering")
	fmt.Println("-----------------------------------------------------")

	versicolorOnly := df.Filter(dataframe.F{
		Colname:    " Species",
		Comparator: "==",
		Comparando: "Iris-versicolor",
	})

	//versicolorOnly = df[df[" Species"] == "Iris-versicolor"]

	fmt.Println(versicolorOnly)

	// Column Selection
	fmt.Println("-----------------------------------------------------")
	fmt.Println("Column Selection")
	fmt.Println("-----------------------------------------------------")

	attrFiltered := df.Select([]string{"Petal length", "Sepal length"})
	fmt.Println(attrFiltered)
}

func SimpleKNNclassifier() {

	// Load the Data
	fmt.Println("Load our csv data")
	rawData, err := base.ParseCSVToInstances("iris_headers.csv", true)
	if err != nil {
		panic(err)
	}

	// Initialize a KNN Classifier
	fmt.Println("Initialize our KNN classifier")
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)

	// Training-Testing Split
	fmt.Println("Perform a training-test split")
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.50)

	// Train the Classifier
	cls.Fit(trainData)

	fmt.Println("Calculate the euclidian distance and return the most popular label")
	predictions, err := cls.Predict(testData)
	if err != nil {
		panic(err)
	}
	fmt.Println(predictions)

	// Summary Metrics
	fmt.Println("Print our summary metrics")
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))

}

func main() {

	Manipulation()

	SimpleKNNclassifier()

}
