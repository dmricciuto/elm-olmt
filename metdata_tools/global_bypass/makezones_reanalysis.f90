program makezones


!Take the CLM CRU-NCEP data and aggregate by time over the spinup period (1901-1920)
!  and "chop" it up into 24 longitundinal zones

use netcdf
implicit none
include 'mpif.h'
!include 'netcdf.h'

integer ng, res
integer v, n, i, z, y,m,j, myid, np
character(len=4) yst, startyrst, endyrst
character(len=4) mst, myidst
character(len=4) zst
character(len=1) rst
character(len=150) metvars, metvars_in, myforcing, myres
character(len=300) fname, filename_base
character(len=300) inputdir, outdir, met_prefix
real data_in(720,360,1460)         !248
integer*2 data_zone(1460,200000)  !2920
integer*2 temp_zone(1460,200000)   !248
real longxy(720,360), latixy(720,360)
real lon(720), lat(360)
real longxy_out(24,200000), latixy_out(24,200000)
integer count_zone(24), ncid_out(24)
integer starti(3), counti(3), dimid(2), starti_out, starti_out_year
integer ierr, ncid, varid, varids_out(24,10), ndaysm(12)
integer mask(720,360), startyear, endyear
double precision dtime(1460)    !2920
real add_offsets(8), scale_factors(8), data_ranges(8,2)
data ndaysm /31,28,31,30,31,30,31,31,30,31,30,31/

ng=62482

call MPI_init(ierr)
call MPI_Comm_rank(MPI_COMM_WORLD, myid, ierr)
call MPI_Comm_size(MPI_COMM_WORLD, np, ierr)
!myid=0
!np=1

!Set the TRENDY CRUJRA input and cpl_bypass output paths.
inputdir = '/projects/hpcl-cli185/proj-shared/xyk/TRENDY2026/processed_inputdata/metdata'
outdir   = '/projects/hpcl-cli185/world-shared/e3sm/inputdata/atm/datm7/atm_forcing.CRUJRA_trendy_2026/cpl_bypass_full'
myforcing = 'elmforc.TRENDY.c2026_0.5x0.5'

!Set the date range and time resolution
startyear = 1901
endyear   = 2025
res       = 6      !Timestep in hours


data_ranges(1,1) =-0.04
data_ranges(1,2) = 0.04   !Precip
data_ranges(2,1) = -20.
data_ranges(2,2) = 2000.  !FSDS
data_ranges(3,1) = 175.
data_ranges(3,2) = 350.   !TBOT
data_ranges(4,1) = 0.
data_ranges(4,2) = 0.10   !QBOT
data_ranges(5,1) = 0.
data_ranges(5,2) = 1000.  !FLDS
data_ranges(6,1) = 20000.
data_ranges(6,2) = 120000.  !PBOT
data_ranges(7,1) = -1.
data_ranges(7,2) = 100.     !WIND
data_ranges(8,1) = -1.
data_ranges(8,2) = 100.     !WIND

do v=1,8
   add_offsets(v) = (data_ranges(v,2)+data_ranges(v,1))/2.
   scale_factors(v) = (data_ranges(v,2)-data_ranges(v,1))*1.1/2**15
end do

mask(:,:)=0
!write(myidst, '(I4)') myid*3
!call system('sleep ' // myidst)
print*, 'myid', myid
if (myid .eq. 0) open(unit=8, file='zone_mappings.txt')

do v=myid+1,7,np
 do z=1,1
   mask(:,:)=0
   if (v .eq. 1) metvars='PRECTmms'
   if (v .eq. 2) metvars='FSDS'
   if (v .eq. 3) metvars='TBOT'
   if (v .eq. 4) metvars='QBOT'
   if (v .eq. 5) metvars='FLDS'
   if (v .eq. 6) metvars='PSRF'
   if (v .eq. 7) metvars='WIND'
   !if (v .eq. 8) metvars='VWIND'

   metvars_in = metvars
   !if (v .eq. 1) metvars_in='pre'
   !if (v .eq. 2) metvars_in='tswrf'
   !if (v .eq. 3) metvars_in='tmp'
   !if (v .eq. 4) metvars_in='spfh'
   !if (v .eq. 5) metvars_in='dlwrf'
   !if (v .eq. 6) metvars_in='pres'
   !if (v .eq. 7) metvars_in='ugrd'
   !if (v .eq. 8) metvars_in='vgrd'


   write(rst,'(I1)') res
   myres = trim(rst) // 'Hrly'

   if (v .eq. 1) met_prefix = trim(inputdir) // '/' // trim(myforcing) // '.PREC.'
   if (v .eq. 2) met_prefix = trim(inputdir) // '/' // trim(myforcing) // '.Solr.'
   if (v .ge. 3) met_prefix = trim(inputdir) // '/' // trim(myforcing) // '.TPQWL.'

   data_in(:,:,:)=1e36
   write(startyrst,'(I4)') startyear
   write(endyrst,'(I4)') endyear
   !z = mod(myid,24)+1
   starti_out = 1

   do y=startyear,endyear
      starti_out_year = 1
      do m=1,1
         write(mst,'(I4)') 1000+m
         count_zone(:)=0
         write(yst,'(I4)') y
         !if (v .eq. 2) then
         fname = trim(met_prefix) // yst // '.nc'
         !else
         !       fname = trim(met_prefix) // yst // '.365d.noc.nc'
         !end if
         print*, fname
         ierr = nf90_open(trim(fname), NF90_NOWRITE, ncid)
         ierr = nf90_inq_varid(ncid, 'LONGXY', varid)
         ierr = nf90_get_var(ncid, varid, lon)
         ierr = nf90_inq_varid(ncid, 'LATIXY', varid)
         ierr = nf90_get_var(ncid, varid, lat)
         ierr = nf90_inq_varid(ncid, trim(metvars_in), varid)
         starti(1:3)  = 1
         starti(1)    = 1
         counti(1)  = 720
         counti(2)  = 360
         counti(3)  = 365*(24/res)

         ierr = nf90_get_var(ncid, varid, data_in(1:counti(1),1:counti(2),1:counti(3)), starti, counti)
         !Precip is mm/6hr, convert to mm/s
         if (v .eq. 1) data_in = data_in / (res*3600.0)
         print*, 'READVAR', y, m, z, v, ierr
         ierr = nf90_close(ncid)
         do i=1,720
            do j=1,360
               if (.not. isnan(data_in(i,j,1)) .and. data_in(i,j,1) < 1e20) mask(i,j)=1
               if (mask(i,j) == 1 .and. lon(i) .ge. -180*z .and. lon(i) .le. 360*z) then
                  count_zone(z) = count_zone(z)+1
                  if (y .eq. startyear .and. m .eq. 1 .and. myid .eq. 0) then
                    write(8,'(2(f10.3,1x),2(I6,1x))') lon(i), lat(j), z, count_zone(z)
                  end if
                  !print*, z,i,j,count_zone(z), data_in(i,j,1:10)
                  temp_zone(1:365*(24/res),count_zone(z)) = &
                    nint((data_in(i,j,1:365*(24/res))-add_offsets(v))/scale_factors(v))
                  !print*, count_zone(z), data_in(i,j,1:10)
                  longxy_out(z, count_zone(z)) = lon(i)
                  latixy_out(z, count_zone(z)) = lat(j)
               end if
            end do
         end do

         !do z=mod(myid,24)+1,24,np
            write(zst,'(I4)') 1000+z
            if (y .eq. startyear .and. m .eq. 1) then
               fname = trim(outdir) // '/' // trim(myforcing) &
                      // '_' // trim(metvars) // '_' // startyrst // '-' // endyrst // '_z' // &
                    zst(3:4) // '.nc'
               ierr = nf90_create(trim(fname),cmode=or(nf90_clobber,nf90_64bit_offset),ncid=ncid_out(z))
               ierr = nf90_def_dim(ncid_out(z), 'n', count_zone(z), dimid(2))
               ierr = nf90_def_dim(ncid_out(z), 'DTIME', (endyear-startyear+1)*(8760/res), dimid(1))
               ierr = nf90_def_var(ncid_out(z), 'DTIME', NF90_DOUBLE, dimid(1), &
                    varids_out(z,1))
               ierr = nf90_put_att(ncid_out(z), varids_out(z,1), 'long_name', &
                    'Day of Year')
               ierr = nf90_put_att(ncid_out(z), varids_out(z,1), 'units', &
                    'Days since ' // startyrst // '-01-01 00:00')
               ierr = nf90_def_var(ncid_out(z), 'LONGXY', NF90_FLOAT, dimid(2), &
                    varids_out(z,2))
               ierr = nf90_def_var(ncid_out(z), 'LATIXY', NF90_FLOAT, dimid(2), &
                    varids_out(z,3))
               ierr = nf90_def_var(ncid_out(z), trim(metvars), NF90_SHORT, &
                    dimid(1:2), varids_out(z,4))
               ierr = nf90_put_att(ncid_out(z), varids_out(z,4), 'add_offset', &
                    add_offsets(v))
               ierr = nf90_put_att(ncid_out(z), varids_out(z,4), 'scale_factor', &
                    scale_factors(v))
               ierr = nf90_enddef(ncid_out(z))
               ierr = nf90_put_var(ncid_out(z), varids_out(z,2), &
                    longxy_out(z, 1:count_zone(z)))
               ierr = nf90_put_var(ncid_out(z), varids_out(z,3), &
                    latixy_out(z, 1:count_zone(z)))
            end if
            do i=1,365*(24/res)
               dtime(i) = (starti_out+i-1)/(24/res*1.0)-(res/24)*0.5
            end do
            starti(1) = starti_out
            counti(1) = 365*(24/res)
            ierr = nf90_put_var(ncid_out(z), varids_out(z,1), &
                 dtime(1:counti(1)), starti(1:1), counti(1:1))
            starti(2) = 1
            counti(2) = count_zone(z)

            data_zone(starti_out_year:(starti_out_year+counti(1)-1),starti(2):(starti(2) &
                 +counti(2)-1)) = temp_zone(1:counti(1),1:counti(2))
            !ierr = nf90_put_var(ncid_out(z), varids_out(z,4), &
            !     temp_zone(1:counti(1), 1:count_zone(z)), starti(1:2), counti(1:2))
            !print*,'WRITEVAR', ierr
         !end do  !Zone loop
         starti_out_year = starti_out_year+365*(24/res)
         starti_out = starti_out+365*(24/res)
         !print*, starti_out
      end do    !month loop
      starti(1) = (y-startyear)*(8760/res)+1
      starti(2) = 1
      counti(1) = 8760/res
      counti(2) = count_zone(z)
      ierr = nf90_put_var(ncid_out(z), varids_out(z,4), &
                   data_zone(1:counti(1), 1:counti(2)), starti(1:2), counti(1:2))
      print*, 'WRITEVAR', y, z, v, starti(1), ierr
   end do       !year loop
   ierr = nf90_close(ncid_out(z))
 end do  !zone loop
 if (myid .eq. 0) close(8)
end do   !Variable loop
call MPI_Finalize(ierr)

end program makezones
