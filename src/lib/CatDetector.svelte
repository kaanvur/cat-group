<script>
	import '@tensorflow/tfjs-backend-webgl';
	import * as cocoSSD from '@tensorflow-models/coco-ssd';
	import { onMount } from 'svelte';
	import { writable } from 'svelte/store';

	const catCategories = ['cat', 'kitten'];
	let model;
	let image;

	// Create a writable store to hold the cat images
	const cats = writable([]);

	async function init() {
		model = await cocoSSD.load();
	}

	async function detectCats() {
		if (!image) {
			console.error('No image to detect cats from');
			return;
		}

		const predictions = await model.detect(image);
		const catPredictions = predictions.filter((prediction) =>
			catCategories.includes(prediction.class)
		);
		return catPredictions;
	}

	function addCat(catPredictions) {
		const groupedCats = groupCats(catPredictions);
		cats.update((values) => [...values, ...groupedCats]);
	}

	function groupCats(catPredictions) {
		const grouped = {};
		catPredictions.forEach((prediction) => {
			const key = `${prediction.bbox[0]}-${prediction.bbox[1]}`;
			if (!grouped[key]) {
				grouped[key] = {
					id: key,
					predictions: []
				};
			}
			grouped[key].predictions.push(prediction);
		});
		return Object.values(grouped);
	}

	function getUniqueId(catPredictions) {
		const uniqueIds = new Set();
		catPredictions.forEach((prediction) => uniqueIds.add(prediction.objectId));
		return Array.from(uniqueIds)[0];
	}

	async function handleFileUpload(event) {
		const file = event.target.files[0];
		const reader = new FileReader();
		reader.onload = async (event) => {
			image = new Image();
			image.src = event.target.result;
			image.onload = async () => {
				const catPredictions = await detectCats();
				addCat(catPredictions);
			};
		};
		reader.readAsDataURL(file);
	}

	onMount(() => {
		init();
	});
	$: console.log($cats);
</script>

<input type="file" accept="image/*" on:change={handleFileUpload} />
