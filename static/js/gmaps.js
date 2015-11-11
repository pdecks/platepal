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

        var resList = $("#results-list-"+cat);
        var infoWindow = new google.maps.InfoWindow();

        for (var i=0; i<top5json[cat].length; i++){
          var biz = top5json[cat][i];
          var letter = String.fromCharCode("A".charCodeAt(0) + i);
          var latLng = new google.maps.LatLng(biz.lat, biz.lng);

          var marker = new google.maps.Marker({
            position: latLng,
            title: biz.name,
            icon: "http://maps.google.com/mapfiles/marker" + letter + ".png"
          });
          
          marker.setMap(map);

          if (cat !== 'gltn'){
            marker.setVisible(false);
          }
          else {
            marker.setVisible(true);

          }

          // create an event handler to listen for marker clicks
          // opens an infoWindow on the marker when clicked
          (function (marker, biz) {
            google.maps.event.addListener(marker, "click", function (e){
              //wrap the content inside an html div to set height and width of InfoWindow
              infoWindow.setContent('<div id="content" style="width:200px;min-height:40px">'+
                '<div id="siteNotice">'+
                '</div>'+
                '<h3 id="firstHeading" class="firstHeading">'+ biz.name + '</h3>'+
                '<div id="bodyContent">'+
                'Average Review by Category: ' + biz.avg_cat_review + '</br>' +
                '</div>'+
                '</div>');
              // infoWindow.setPosition()
              infoWindow.open(map, marker);
            });
          }) (marker, biz); // TODO: understand this

          markers[cat].push(marker);

          resList.append("<li>" + biz.name + " " + biz.avg_cat_review + "</li>");
        } // end inner for loop over businesses in list by category
                
      } // end outer for loop over categories
      
  }); // end $.getJSON

} // end setMarkers

// create listeners by category button


function filterResults(catID) {
  // slice off the 4 letter cat code
  cat = catID.slice(0,4);

  // turn on markers in current cat
  // show list for current cat
  for (var j in markers[cat]){
    markers[cat][j].setVisible(true);
  }
  var name = "div#results-"+cat;
  $(name).removeClass("hidden");
  // styleAttr = $("div.results-"+cat).attr("style");
  // console.log('this is stleAttr for ' + cat);
  console.log('"'+'div#results-'+cat+'"');
  // console.log(styleAttr);

  // turn off markers in other cats
  // hide lists for other cats
  for (var i = 0; i < catCodes.length; i++){
    var currentCat = catCodes[i];

    if (currentCat !== cat){
      name = "div#results-"+currentCat;
      $(name).addClass("hidden");
      for (var k in markers[currentCat]){
        markers[currentCat][k].setVisible(false);
      } // end inner for
    } // end if
  } // end for 
} // end filterResults



// create click listener on all links in map navigation bar
$("a.map-nav").on('click', function(evt){
  // get the link's id
  var catID = $(this).attr('id');
  // filter the map results based on the category
  filterResults(catID);

});

$(document).ready(initMap());

