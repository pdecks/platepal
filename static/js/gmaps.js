// define arrays for the markers by category
var markers = new Array();

var catCodes = ['gltn', 'vgan', 'kshr', 'algy', 'pleo'];

function initMap(){
  // update to geolocate or set default SF
  // var myLatLng = {lat: 37.754407, lng: -122.447684}; // SF
  var myLatLng = {lat: 39.7392, lng: -104.9903}; // Denver, CO
  
  // define map
  var map = new google.maps.Map(document.getElementById('map'),{
      center: myLatLng,
      zoom: 4,
  });
  
  // define markers
  setMarkers(map);

  var infoWindow = new google.maps.InfoWindow({map: map});

   // Try HTML5 geolocation.
  if (navigator.geolocation) {
    navigator.geolocation.getCurrentPosition(function(position) {
      var pos = {
        lat: position.coords.latitude,
        lng: position.coords.longitude
      };

      infoWindow.setPosition(pos);
      infoWindow.setContent('Location found.');
      // map.setCenter(pos);
    }, function() {
      handleLocationError(true, infoWindow, map.getCenter());
    });
  } else {
  // Browser doesn't support Geolocation
    handleLocationError(false, infoWindow, map.getCenter());
  }

} // end initMap


function handleLocationError(browserHasGeolocation, infoWindow, pos) {
  infoWindow.setPosition(pos);
  infoWindow.setContent(browserHasGeolocation ?
                        'Error: The Geolocation service failed.' :
                        'Error: Your browser doesn\'t support geolocation.');
}

// JSON = {'gltn': [{'biz_id': biz_id, 'avg_cat_review': avg_cat_review, 'lat': lat, 'lng': lng}, {}, {}],
//         'vgan': [{}, {}, {}],
//          ...
//         }

function setMarkers(map) {
// Adds markers to the map.

  $.get("/popular-biz.json", function(top5json) {
      console.log('in $.get');
      console.log(top5json);
      // iterate over each item in the dictionary with $.each()
      for (var cat in top5json) {
        markers[cat] = new Array();

        var ol = $("#results-list-"+cat);

        for (var i=0; i<top5json[cat].length; i++){
          var biz = top5json[cat][i];

          var latLng = new google.maps.LatLng(biz.lat, biz.lng);

          var marker = new google.maps.Marker({
            position: latLng,
            title: biz.name,
            
          });
          // marker.setMap(map);
          if (cat !== 'gltn'){
            marker.setMap(null);
          }
          else {
            marker.setMap(map);
          }
          markers[cat].push(marker);

          ol.append("<li>" + biz.name + " " + biz.avg_cat_review + "</li>");
        } // end inner for loop over businesses in list by category
                
      } // end outer for loop over categories

      // clear all markers TODO: fix function, not working
      // clearMarkers();
      
      // category = 'gltn';
      // add only markers for GF (default) todo: fix, not working
      // showMarkersByCategory(category, Map);

      
  }); // end $.getJSON

} // end setMarkers

// create listeners by category button

function showMarkersByCategory(cat, map){
  console.log('in showMarkersByCategory');
  for (var i = 0; i < markers[cat].length; i++){
    console.log(markers[cat]);
    markers[cat][i].setMap(map);
  }
}

function clearMarkers() {
  setMapOnAll(null);
}

function setMapOnAll(map){
  for (var i = 0; i < markers.length; i++){
    cat = catCodes[i];
    for (var j = 0; j < markers[cat].length; j++){
      console.log('this is markers[cat][j]: ');
      console.log(markers[cat][j]);
      markers[cat][j].setMap(map);
    }
  }
}

function flakySubmit(evt) {
  evt.preventDefault();
  // add call to show markers
  alert('In flakySubmit');
}


// create arrays by category
// when the category button is 'submitted', add those markers 
function categoryMarkersOn(evt) {

}

var gltnButton = document.getElementById("gltn-map-filter");
var vganButton = document.getElementById("vgan-map-filter");
var kshrButton = document.getElementById("kshr-map-filter");
var algyButton = document.getElementById("algy-map-filter");
var pleoButton = document.getElementById("pleo-map-filter");

gltnButton.addEventListener('submit', flakySubmit);
vganButton.addEventListener('submit', flakySubmit);
kshrButton.addEventListener('submit', flakySubmit);
algyButton.addEventListener('submit', flakySubmit);
pleoButton.addEventListener('submit', flakySubmit);
// first solved on button click, then updated to form submit


 // id="gluten-free-map-filter">

$(document).ready(initMap());

// TODO:
// hide the other categories' markers (GF is default)
// number markers


// $.EACH VERSION
// $(document).ready(function(){
//   $.getJSON("/popular-biz.json", function(top5json) {
//       console.log(top5json);
//       // iterate over each item in the dictionary with $.each()
//       $.each(top5json, function(cat, catList) {
     
//         console.log(cat);
//         console.log(catList);

//         var bizId;
//         // add business to list to display alongside map
//         var ul = $("#results-list");
//         var marker;
//         // extract top 5 businesses --> has as a for loop before
//         $.each(catList, function(idx, biz) {
//           bizId = biz.biz_id;
          
//           var latLng = new google.maps.LatLng(biz.lat, biz.lng);
//           console.log(latLng);
//           console.log(biz.lat);
//           console.log(biz.lng);

//           // Creating a marker and putting it on the map
//           marker = new google.maps.Marker({
              
//               position: latLng,
//               title: biz.name,
//               map: map,

//           });

//           // marker.setMap(map);
//           console.log("This is after var marker");
//           console.log(marker.position);
//           console.log(map);
//           ul.append("<li>" + biz.name + " " + biz.avg_cat_review + "</li>");
//         }); // end $.each(catList)
    
//       }); // end $.each(top5json)

//   }); // end $.getJSON

// }); // end $(document)

// // // individual marker
// // var marker = new google.maps.Marker({
// //     position = myLatlng,
// //     map: map,
// //     title: 'Hover text',
// //     icon: '(optional to define custom icon image)'
// // });


// // Data for the markers consisting of a name, a LatLng and a zIndex for the
// // order in which these markers should display on top of each other.
// var searchResults = [
//   ['Haight Shrader', 37.769387, -122.451875, 4],
//   ['Divis Page', 37.772223, -122.437159, 5],
//   ['Valencia Mission', 37.745328, -122.420057, 3],
//   ['Glen Park', 37.734511, -122.433894, 2],
//   ['West Portal', 37.740884, -122.465754, 1]
// ];


  // info for custom markers...
  // Origins, anchor positions and coordinates of the marker increase in the X
  // direction to the right and in the Y direction down.
  // var image = {
  //   url: 'images/beachflag.png',
  //   // This marker is 20 pixels wide by 32 pixels high.
  //   size: new google.maps.Size(20, 32),
  //   // The origin for this image is (0, 0).
  //   origin: new google.maps.Point(0, 0),
  //   // The anchor for this image is the base of the flagpole at (0, 32).
  //   anchor: new google.maps.Point(0, 32)
  // };
  // Shapes define the clickable region of the icon. The type defines an HTML
  // <area> element 'poly' which traces out a polygon as a series of X,Y points.
  // The final coordinate closes the poly by connecting to the first coordinate.
  // var shape = {
  //   coords: [1, 1, 1, 20, 18, 20, 18, 1],
  //   type: 'poly'
  // };