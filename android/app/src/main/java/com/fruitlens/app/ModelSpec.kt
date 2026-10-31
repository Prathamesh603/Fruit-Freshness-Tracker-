package com.fruitlens.app

import org.json.JSONObject

data class FruitLabel(val id: String, val fruit: String, val condition: String, val displayName: String)

data class ModelSpec(
    val modelId: String,
    val modelFile: String,
    val sha256: String,
    val width: Int,
    val height: Int,
    val interpolation: String,
    val mean: FloatArray,
    val std: FloatArray,
    val scoreType: String,
    val labels: List<FruitLabel>,
    val minScore: Float,
    val thresholdValidated: Boolean,
) {
    companion object {
        fun parse(json: String): ModelSpec {
            val root = JSONObject(json)
            require(root.getInt("schemaVersion") == 1) { "Unsupported model manifest version." }
            val input = root.getJSONObject("input")
            require(input.getString("layout") == "NHWC" && input.getInt("channels") == 3)
            require(input.getString("colorSpace") == "RGB" && input.getString("resizeMode") == "stretch")
            val width = input.getInt("width")
            val height = input.getInt("height")
            require(width in 16..1024 && height in 16..1024) { "Invalid model dimensions." }
            val interpolation = input.optString("interpolation", "bilinear")
            require(interpolation in setOf("bilinear", "nearest")) { "Unsupported resize interpolation." }
            fun channels(name: String): FloatArray {
                val array = input.getJSONArray(name)
                require(array.length() == 3)
                return FloatArray(3) { array.getDouble(it).toFloat().also { v -> require(v.isFinite()) } }
            }
            val mean = channels("mean")
            val std = channels("std").also { require(it.all { v -> v > 0 }) }
            val output = root.getJSONObject("output").getString("scoreType")
            require(output in setOf("logits", "probabilities"))
            val rawLabels = root.getJSONArray("labels")
            require(rawLabels.length() in 2..100)
            val labels = List(rawLabels.length()) { index ->
                val label = rawLabels.getJSONObject(index)
                FruitLabel(label.getString("id"), label.getString("fruit"),
                    label.getString("condition"), label.getString("displayName")).also {
                    require(listOf(it.id, it.fruit, it.condition, it.displayName).all(String::isNotBlank))
                }
            }
            require(labels.map { it.id }.distinct().size == labels.size)
            val decision = root.getJSONObject("decision")
            val threshold = decision.getDouble("minScore").toFloat()
            require(threshold.isFinite() && threshold in 0f..1f)
            val file = root.getString("modelFile")
            require(file == "freshness.tflite") { "Expected freshness.tflite in the bundle." }
            val hash = root.getString("sha256").lowercase()
            require(hash.matches(Regex("[a-f0-9]{64}"))) { "Model hash is invalid." }
            return ModelSpec(root.getString("modelId"), file, hash, width, height, interpolation, mean, std,
                output, labels, threshold, decision.getBoolean("thresholdValidated"))
        }
    }
}
