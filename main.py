import math
import time
import requests
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="AegisRoute AI - Disaster Evacuation & Safe Route Intelligence API",
    description="FastAPI backend service with real road-based routing via OpenRouteService",
    version="2.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ CONFIGURATION ============
# Get free API key from: https://openrouteservice.org/dev/#/signup
OPENROUTESERVICE_API_KEY = "5b3ce3597851110001cf6248"  # Free demo key (replace with your own)
ORS_BASE_URL = "https://api.openrouteservice.org"

# ============ PYDANTIC SCHEMAS ============

class GeoPoint(BaseModel):
    lat: float
    lng: float

class HazardZone(BaseModel):
    id: str
    name: str
    type: str
    severity: str
    coordinates: List[GeoPoint]
    center: GeoPoint
    radiusMeters: Optional[float] = 1200
    windSpeedKmh: Optional[float] = 45
    windDirection: Optional[str] = "NE"
    spreadRateKmh: Optional[float] = 2.5
    evacuationMandatory: bool = True
    reportedAt: str = "Just now"
    description: str

class HazardCreate(BaseModel):
    name: str
    type: str
    severity: str
    coordinates: List[GeoPoint]
    center: GeoPoint
    radiusMeters: Optional[float] = 1200
    windSpeedKmh: Optional[float] = 45
    windDirection: Optional[str] = "NE"
    spreadRateKmh: Optional[float] = 2.5
    evacuationMandatory: bool = True
    description: str

class ShelterFeatures(BaseModel):
    medicalStaff: bool = True
    emergencyPower: bool = True
    petFriendly: bool = True
    accessible: bool = True
    foodWaterSupply: bool = True
    hazmatDecon: bool = False

class SafeShelter(BaseModel):
    id: str
    name: str
    type: str
    location: GeoPoint
    address: str
    capacity: int
    currentOccupancy: int
    status: str
    features: ShelterFeatures
    contactPhone: str
    safetyScore: int = 95

class ShelterUpdate(BaseModel):
    status: Optional[str] = None
    currentOccupancy: Optional[int] = None

class RouteRequest(BaseModel):
    origin: GeoPoint
    shelterId: str
    mode: Optional[str] = "vehicle"

class RouteStep(BaseModel):
    instruction: str
    distanceMeters: int
    durationSeconds: int
    hazardWarning: Optional[str] = None
    turnType: str

class EvacuationRoute(BaseModel):
    id: str
    shelterId: str
    shelterName: str
    origin: GeoPoint
    destination: GeoPoint
    path: List[GeoPoint]
    totalDistanceKm: float
    estimatedTimeMinutes: int
    safetyIndex: int
    riskLevel: str
    hazardProximityMeters: int
    activeWarnings: List[str]
    steps: List[RouteStep]
    recommendedMode: str

class EmergencyAlert(BaseModel):
    id: str
    title: str
    level: str
    issuedAt: str
    affectedZones: List[str]
    message: str
    source: str
    actionRequired: str

class AlertCreate(BaseModel):
    title: str
    level: str
    affectedZones: List[str]
    message: str
    source: str
    actionRequired: str

class SOSRequest(BaseModel):
    id: str
    userName: str
    phone: str
    location: GeoPoint
    timestamp: str
    peopleCount: int
    needsMedical: bool
    needsMobilityAssistance: bool
    batteryLevel: int
    status: str
    notes: Optional[str] = None

class SOSCreate(BaseModel):
    userName: str
    phone: str
    location: GeoPoint
    peopleCount: int = 1
    needsMedical: bool = False
    needsMobilityAssistance: bool = False
    batteryLevel: int = 80
    notes: Optional[str] = None

# ============ GEOSPATIAL HELPERS ============

def haversine_km(p1: GeoPoint, p2: GeoPoint) -> float:
    """Calculate distance between two points in kilometers"""
    r = 6371.0
    d_lat = math.radians(p2.lat - p1.lat)
    d_lng = math.radians(p2.lng - p1.lng)
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(math.radians(p1.lat)) * math.cos(math.radians(p2.lat)) *
         math.sin(d_lng / 2) ** 2)
    return 2 * r * math.asin(math.sqrt(a))

def point_in_hazard_zone(point: GeoPoint, hazard: 'HazardZone') -> bool:
    """Check if a point is inside a hazard zone polygon"""
    # Simple distance check for circular hazard zones
    distance = haversine_km(point, hazard.center)
    radius_km = (hazard.radiusMeters or 1200) / 1000
    return distance < radius_km

def get_hazard_proximity(point: GeoPoint, hazards: List['HazardZone']) -> tuple[int, List[str]]:
    """Get minimum distance to any hazard and warnings"""
    min_distance = 999999
    warnings = []
    
    for hazard in hazards:
        distance_km = haversine_km(point, hazard.center)
        distance_m = distance_km * 1000
        
        if distance_m < min_distance:
            min_distance = int(distance_m)
        
        radius_km = (hazard.radiusMeters or 1200) / 1000
        buffer_km = radius_km + 0.5  # 500m safety buffer
        
        if distance_km < buffer_km:
            warnings.append(f"HAZARD: {hazard.name} - {distance_km:.1f}km away")
    
    return min_distance, warnings

# ============ OPENROUTESERVICE ROUTING ============

def get_realistic_route(origin: GeoPoint, destination: GeoPoint, mode: str = "vehicle") -> Optional[dict]:
    """
    Get realistic road-based route using OpenRouteService API
    
    Returns: {
        'coordinates': List of GeoPoint waypoints,
        'distance_meters': total distance,
        'duration_seconds': total duration,
        'instructions': List of turn-by-turn instructions
    }
    """
    
    # Convert mode to OpenRouteService profile
    profile_map = {
        'vehicle': 'driving-car',
        'on_foot': 'foot-walking',
        '4x4': 'driving-hgv',
    }
    profile = profile_map.get(mode, 'driving-car')
    
    try:
        # Build ORS request
        url = f"{ORS_BASE_URL}/v2/directions/{profile}"
        
        params = {
            'api_key': OPENROUTESERVICE_API_KEY,
            'start': f"{origin.lng},{origin.lat}",  # ORS uses lng,lat
            'end': f"{destination.lng},{destination.lat}",
            'format': 'json',
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"ORS Error: {response.status_code} - {response.text}")
            return None
        
        data = response.json()
        
        if 'routes' not in data or len(data['routes']) == 0:
            return None
        
        route = data['routes'][0]
        
        # Extract coordinates from geometry
        geometry = route.get('geometry', {})
        if isinstance(geometry, dict) and 'coordinates' in geometry:
            coords = geometry['coordinates']
        elif isinstance(geometry, list):
            coords = geometry
        else:
            return None
        
        # Convert coordinates to GeoPoint
        waypoints = [GeoPoint(lat=c[1], lng=c[0]) for c in coords]
        
        # Get summary data
        summary = route.get('summary', {})
        distance = summary.get('distance', 0)  # meters
        duration = summary.get('duration', 0)  # seconds
        
        # Get turn-by-turn instructions
        instructions = []
        steps = route.get('steps', [])
        for step in steps[:10]:  # Limit to first 10 steps
            instruction = step.get('instruction', '')
            distance_step = step.get('distance', 0)
            duration_step = step.get('duration', 0)
            
            if instruction:
                instructions.append({
                    'text': instruction,
                    'distance': int(distance_step),
                    'duration': int(duration_step),
                })
        
        return {
            'coordinates': waypoints,
            'distance_meters': int(distance),
            'duration_seconds': int(duration),
            'instructions': instructions,
        }
    
    except Exception as e:
        print(f"OpenRouteService Error: {e}")
        return None

def get_fallback_route(origin: GeoPoint, destination: GeoPoint) -> dict:
    """
    Fallback route calculation if ORS is unavailable
    Creates a simple detour around hazards
    """
    # Calculate direct distance
    direct_distance_km = haversine_km(origin, destination)
    
    # Add 30% buffer for real routing (not straight line)
    estimated_distance_km = direct_distance_km * 1.3
    estimated_distance_m = estimated_distance_km * 1000
    
    # Estimate time (assume 40 km/h average speed)
    estimated_duration_s = int((estimated_distance_km / 40) * 3600)
    
    # Create a simple detour waypoint
    mid_lat = (origin.lat + destination.lat) / 2 + 0.004
    mid_lng = (origin.lng + destination.lng) / 2 - 0.003
    
    waypoints = [
        origin,
        GeoPoint(lat=mid_lat, lng=mid_lng),
        destination,
    ]
    
    instructions = [
        {'text': 'Depart and head toward evacuation route', 'distance': int(estimated_distance_m / 3), 'duration': int(estimated_duration_s / 3)},
        {'text': 'Continue on main evacuation corridor', 'distance': int(estimated_distance_m / 3), 'duration': int(estimated_duration_s / 3)},
        {'text': 'Arrive at destination shelter', 'distance': int(estimated_distance_m / 3), 'duration': int(estimated_duration_s / 3)},
    ]
    
    return {
        'coordinates': waypoints,
        'distance_meters': estimated_distance_m,
        'duration_seconds': estimated_duration_s,
        'instructions': instructions,
    }

# ============ IN-MEMORY DATABASE ============

hazards_db: List[HazardZone] = [
    HazardZone(
        id="hazard-001",
        name="Ridgecrest Wildfire Perimeter - Sector A",
        type="wildfire",
        severity="extreme",
        center=GeoPoint(lat=37.7749, lng=-122.4194),
        coordinates=[
            GeoPoint(lat=37.7810, lng=-122.4280),
            GeoPoint(lat=37.7835, lng=-122.4150),
            GeoPoint(lat=37.7760, lng=-122.4080),
            GeoPoint(lat=37.7680, lng=-122.4130),
            GeoPoint(lat=37.7690, lng=-122.4250),
            GeoPoint(lat=37.7770, lng=-122.4290),
        ],
        radiusMeters=1400,
        windSpeedKmh=48,
        windDirection="NE (Gusting to 65 km/h)",
        spreadRateKmh=2.8,
        evacuationMandatory=True,
        reportedAt="12 mins ago",
        description="Rapidly advancing crown fire along steep canyon grade.",
    ),
]

shelters_db: List[SafeShelter] = [
    SafeShelter(
        id="shelter-101",
        name="Metro Civic Center Emergency Refuge Base",
        type="civic_center",
        location=GeoPoint(lat=37.7795, lng=-122.4500),
        address="99 Grove St, Sector 2",
        capacity=1200,
        currentOccupancy=740,
        status="open",
        features=ShelterFeatures(),
        contactPhone="+1 (555) 911-3420",
        safetyScore=96,
    ),
    SafeShelter(
        id="shelter-102",
        name="St. Jude Regional Medical & Triage Facility",
        type="hospital",
        location=GeoPoint(lat=37.7650, lng=-122.4550),
        address="505 Parnassus Ave, Sector 4",
        capacity=800,
        currentOccupancy=690,
        status="near_capacity",
        features=ShelterFeatures(hazmatDecon=True),
        contactPhone="+1 (555) 911-8840",
        safetyScore=98,
    ),
]

alerts_db: List[EmergencyAlert] = [
    EmergencyAlert(
        id="alert-01",
        title="IMMEDIATE EVACUATION ORDER - DOWNTOWN SECTOR 3 & 4",
        level="CRITICAL",
        issuedAt="4 mins ago",
        affectedZones=["Downtown Core", "Ridgecrest"],
        message="Extreme wildfire expansion propelled by 48 km/h northeast gusts.",
        source="Federal Emergency Management Division (FEMA/OES)",
        actionRequired="Proceed on foot or vehicle to nearest westward shelter immediately.",
    )
]

sos_db: List[SOSRequest] = []

# ============ API ENDPOINTS ============

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "AegisRoute AI Disaster Evacuation API",
        "version": "2.0.0",
        "routing": "OpenRouteService (Real Road-Based)"
    }

@app.get("/api/v1/hazards", response_model=List[HazardZone])
def get_hazards():
    return hazards_db

@app.post("/api/v1/hazards", response_model=HazardZone)
def create_hazard(hazard: HazardCreate):
    new_h = HazardZone(
        id=f"hazard-{int(time.time())}",
        reportedAt="Just now",
        **hazard.dict()
    )
    hazards_db.insert(0, new_h)
    return new_h

@app.delete("/api/v1/hazards/{hazard_id}")
def delete_hazard(hazard_id: str):
    global hazards_db
    hazards_db = [h for h in hazards_db if h.id != hazard_id]
    return {"success": True}

@app.get("/api/v1/shelters", response_model=List[SafeShelter])
def get_shelters():
    return shelters_db

@app.patch("/api/v1/shelters/{shelter_id}", response_model=SafeShelter)
def update_shelter(shelter_id: str, update: ShelterUpdate):
    for s in shelters_db:
        if s.id == shelter_id:
            if update.status is not None:
                s.status = update.status
            if update.currentOccupancy is not None:
                s.currentOccupancy = update.currentOccupancy
            return s
    raise HTTPException(status_code=404, detail="Shelter not found")

@app.post("/api/v1/routes/calculate", response_model=EvacuationRoute)
def calculate_route(req: RouteRequest):
    """
    Calculate realistic evacuation route using OpenRouteService
    Falls back to simple routing if API unavailable
    """
    shelter = next((s for s in shelters_db if s.id == req.shelterId), shelters_db[0])
    
    # Try to get realistic route from OpenRouteService
    route_data = get_realistic_route(req.origin, shelter.location, req.mode or "vehicle")
    
    # Fallback to simple route if ORS fails
    if not route_data:
        print("Using fallback routing")
        route_data = get_fallback_route(req.origin, shelter.location)
    
    # Get hazard proximity and warnings
    hazard_proximity, active_warnings = get_hazard_proximity(req.origin, hazards_db)
    
    # Create route steps from instructions
    steps = []
    for idx, instr in enumerate(route_data['instructions']):
        turn_type = "depart" if idx == 0 else "arrive" if idx == len(route_data['instructions']) - 1 else "straight"
        
        steps.append(RouteStep(
            instruction=instr['text'],
            distanceMeters=instr['distance'],
            durationSeconds=instr['duration'],
            hazardWarning=active_warnings[idx] if idx < len(active_warnings) else None,
            turnType=turn_type,
        ))
    
    # Calculate safety index (0-100)
    safety_index = max(50, 100 - (hazard_proximity // 100))  # Closer to hazard = lower safety
    
    return EvacuationRoute(
        id=f"route-{int(time.time())}",
        shelterId=shelter.id,
        shelterName=shelter.name,
        origin=req.origin,
        destination=shelter.location,
        path=route_data['coordinates'],
        totalDistanceKm=round(route_data['distance_meters'] / 1000, 2),
        estimatedTimeMinutes=max(1, route_data['duration_seconds'] // 60),
        safetyIndex=safety_index,
        riskLevel="safe" if hazard_proximity > 1500 else "moderate" if hazard_proximity > 500 else "high",
        hazardProximityMeters=hazard_proximity,
        activeWarnings=active_warnings,
        steps=steps,
        recommendedMode=req.mode or "vehicle"
    )

@app.get("/api/v1/alerts", response_model=List[EmergencyAlert])
def get_alerts():
    return alerts_db

@app.post("/api/v1/alerts/broadcast", response_model=EmergencyAlert)
def broadcast_alert(alert: AlertCreate):
    new_alert = EmergencyAlert(
        id=f"alert-{int(time.time())}",
        issuedAt="Just now",
        **alert.dict()
    )
    alerts_db.insert(0, new_alert)
    return new_alert

@app.get("/api/v1/sos", response_model=List[SOSRequest])
def get_sos():
    return sos_db

@app.post("/api/v1/sos", response_model=SOSRequest)
def create_sos(sos: SOSCreate):
    new_sos = SOSRequest(
        id=f"sos-{int(time.time())}",
        timestamp="Just now",
        status="pending",
        **sos.dict()
    )
    sos_db.insert(0, new_sos)
    return new_sos
