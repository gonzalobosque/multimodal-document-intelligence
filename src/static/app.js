const classifyForm = document.getElementById("classify-form");
const classificationResult = document.getElementById("classification-result");

const searchForm = document.getElementById("search-form");
const searchResults = document.getElementById("search-results");


async function getErrorMessage(response, fallbackMessage) {
    try {
        const data = await response.json();
        return data.detail || fallbackMessage;
    } catch {
        return fallbackMessage;
    }
}


classifyForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const submitButton = classifyForm.querySelector('button[type="submit"]');
    submitButton.disabled = true;

    classificationResult.textContent = "Processing document...";

    try {
        const formData = new FormData(classifyForm);

        const response = await fetch("/classify", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const message = await getErrorMessage(
                response,
                "Document classification failed."
            );
            throw new Error(message);
        }

        const result = await response.json();

        classificationResult.textContent =
            "File: " + result.filename + "\n" +
            "Class: " + result.predicted_class + "\n" +
            "Confidence: " + (result.confidence * 100).toFixed(1) + "%\n\n" +
            "OCR excerpt:\n" + result.text_excerpt;

    } catch (error) {
        classificationResult.textContent =
            "Error: " + error.message;

    } finally {
        submitButton.disabled = false;
    }
});


searchForm.addEventListener("submit", async function (event) {
    event.preventDefault();

    const submitButton = searchForm.querySelector('button[type="submit"]');
    submitButton.disabled = true;

    searchResults.textContent = "Searching documents...";

    try {
        const formData = new FormData(searchForm);

        const response = await fetch("/search", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const message = await getErrorMessage(
                response,
                "Document search failed."
            );
            throw new Error(message);
        }

        const data = await response.json();

        if (data.results.length === 0) {
            searchResults.textContent = "No documents found.";
            return;
        }

        let resultsText = "";

        data.results.forEach(function (result, index) {
            const documentClass =
                result.predicted_class ??
                result.label_name ??
                "Unclassified";

            resultsText +=
                (index + 1) + ". " + result.filename + "\n" +
                "Class: " + documentClass + "\n";

            if (result.similarity !== null) {
                resultsText +=
                    "Similarity: " + result.similarity.toFixed(3) + "\n";
            }

            resultsText +=
                result.text_excerpt + "\n\n";
        });

        searchResults.textContent = resultsText;

    } catch (error) {
        searchResults.textContent =
            "Error: " + error.message;

    } finally {
        submitButton.disabled = false;
    }
});