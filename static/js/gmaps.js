// var map;
// var markers = new Array([]);

// var coords_1 = [
//     new google.maps.LatLng(60.32522, 19.07002),
//     new google.maps.LatLng(60.45522, 19.12002),
//     new google.maps.LatLng(60.86522, 19.35002),
//     new google.maps.LatLng(60.77522, 19.88002),
//     new google.maps.LatLng(60.36344, 19.36346),
//     new google.maps.LatLng(60.56562, 19.33002)];

// var coords_2 = [
//     new google.maps.LatLng(59.32522, 18.07002),
//     new google.maps.LatLng(59.45522, 18.12002),
//     new google.maps.LatLng(59.86522, 18.35002),
//     new google.maps.LatLng(59.77522, 18.88002),
//     new google.maps.LatLng(59.36344, 18.36346),
//     new google.maps.LatLng(59.56562, 18.33002)];


// function initialize() {
//     console.log("in initialize");

//     var myLatLng = {lat: 37.754407, lng: -122.447684};
  
//     // define map
//     var map = new google.maps.Map(document.getElementById('map'),{
//         center: myLatLng,
//         zoom: 4,
//     });

//     $('button').on('click', function() {

//         if ($(this).data('action') === 'add') {

//             addMarkers($(this).data('filtertype'));

//         } else {

//             removeMarkers($(this).data('filtertype'));
//         }
//     });
// }

// function addMarkers(filterType) {

//     var temp = filterType === 'coords_1' ? coords_1 : coords_2;

//     markers[filterType] = new Array([]);

//     for (var i = 0; i < temp.length; i++) {

//         var marker = new google.maps.Marker({
//             map: map,
//             position: temp[i]
//         });

//         markers[filterType].push(marker);
//     }
// }

// function removeMarkers(filterType) {

//     for (var i = 0; i < markers[filterType].length; i++) {

//         markers[filterType][i].setMap(null);
//     }
// }

// initialize();


function initMap(){
  // update to geolocate or set default SF
  var myLatLng = {lat: 37.754407, lng: -122.447684};
  
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
      map.setCenter(pos);
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


$(document).ready(initMap());

function setMarkers(map) {
// Adds markers to the map.

  $.get("/popular-biz.json", function(top5json) {
      console.log('in $.get');
      console.log(top5json);
      // iterate over each item in the dictionary with $.each()
      for (var cat in top5json) {

        for (var i=0; i<top5json[cat].length; i++){
          var biz = top5json[cat][i];

          var latLng = new google.maps.LatLng(biz.lat, biz.lng);

          var marker = new google.maps.Marker({
            position: latLng,
            title: biz.name,
            
          });
          marker.setMap(map);

              
        } // end inner for loop over businesses in list by category
                
      } // end outer for loop over categories

  }); // end $.getJSON

} // end setMarkers

// JSON = {'gltn': [{'biz_id': biz_id, 'avg_cat_review': avg_cat_review, 'lat': lat, 'lng': lng}, {}, {}],
//         'vgan': [{}, {}, {}],
//          ...
//         }

$(document).ready(function(){
  $.get("/popular-biz.json", function(top5json) {
      // iterate over each item in the dictionary with $.each()
      for (var cat in top5json) {
        // grab ordered list to populate
        var ol = $("#results-list-"+cat);

        // iterate over each business in the category
        for (var i=0; i<top5json[cat].length; i++){
          var biz = top5json[cat][i];

          // add the business info to the appropriate category list
          ol.append("<li>" + biz.name + " " + biz.avg_cat_review + "</li>");
     
        } // end inner for loop over businesses in list by category
            
      } // end outer for loop over categories

  }); // end $.getJSON

}); // end $(document)





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